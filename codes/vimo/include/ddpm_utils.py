import torch

curr_alpha_bar = None # in-scipt copy to avoid pass-in param for sampling
curr_betas = None

def check_betas(betas):
    if betas == None:
        if curr_betas == None:
            raise RuntimeError("betas => noise per timestep vector not provided. To cache it, call 'make_beta_schedule()'")
    else:
        update_alphas(betas)

def update_alphas(betas):
    global curr_betas
    curr_betas = betas
    alphas = 1.0 - betas # ~appox [1-0]
    alphas_cumprod = torch.cumprod(alphas, dim=0)  # \bar{alpha}_t => cummulative product
    global curr_alpha_bar
    curr_alpha_bar = alphas_cumprod

# ------------------ Helper Utility ------------------
def make_beta_schedule(steps, start=1e-4, end=0.02, device='cpu'):
    """
    betas --> represent amount of noise to add at each step
    default:
        Small betas (start=0.0001 → end=0.02) means: 
        each step adds only a tiny amount of noise
    aplhas --> 1-beta, i.e. amount of original input to retain from prev step
    a_bar --> alphas cummulative product
    pad --> ensures to include [t=0]:1 (fully inp) and [t=T-1]:1 (fully noise) into alpha_cum
            torch.cat((torch.tensor([1]), alphas_cumprod, torch.tensor([0]))) --> removed
    """
    betas = torch.linspace(start, end, steps, dtype=torch.float32, device=device) # linear schedule (adjust if needed)
    update_alphas(betas)
    return betas

# ------------------ Helper Utility ------------------
def forward_diffusion_sample(x0, t, betas=None):
    """
    Forward Noising: q_sample: x_t = sqrt(alpha_bar) * x0 + sqrt(1-alpha_bar) * noise
    Input Parameters:
        x0: ground truth input at t=0, here motion data (B, S, D)
        t: (B,) different timesteps for each item in batch, long tensor in [0..T-1]
        betas: noise per timestep tensor (must, if make_beta_schedule() wasn't previously called)
    """
    check_betas(betas)
    betas = curr_betas
    alphas_cumprod = curr_alpha_bar
    
    noise = torch.randn_like(x0) # same shape as x0 or None (then sampled)
    a_bar_batch = alphas_cumprod[t] # gather alpha_bar for each batch
    a_bar_scaled = a_bar_batch.view(-1, 1, 1).to(x0.device)  # (B,1,1)
    #-1 means "infer this dimension from the length of the input".
    # The two 1s add singleton dimensions, making it easy to broadcast in later operations.

    xt = torch.sqrt(a_bar_scaled) * x0 + torch.sqrt(1.0 - a_bar_scaled) * noise
    return xt

# ------------------ Helper Utility ------------------
@torch.no_grad()
def reverse_denoising_sample(model, x_T=(1,150,151), T_steps=1000,  p2d=None, stochasticity=False, betas=None):
    """
    Sequential denoising => x0_hat (deterministic mean) test (B,S,D)
    Input Parameters:
        x_T: (B,S,D) initial noisy sample at t=T-1 (we can sample x_T by q_sample(x0, T-1))
        T_steps: no. of time steps to denoise
        p2d: (B,S,...) 2d poses / condition vectors
        stochasticity: 
            if True: 
                - diverse outputs from the same condition (Creative generation)
                - x_t_minus_1 = μ + sigma_t * z (Original DDPM)
            else False:
                - to reproducible results (same input → same output)
                - sampling process to be non-Markovian
        betas: noise per timestep tensor (must, if make_beta_schedule() wasn't previously called)
    """
    check_betas(betas)
    betas = curr_betas
    alphas_cumprod = curr_alpha_bar
    
    device = model.device
    if isinstance(x_T, tuple):
        x_t = torch.randn(x_t, dtype=float32, device=device) # randn = Gaussian noise, rand = Uniform noise
    elif torch.is_tensor(x_T):
        x_t = x_T.clone().to(device)
    else:
        raise RuntimeError(f"Given format for input x_T is unrecognized. Type: {type(x_T)}")
    
    if stochasticity and not torch.is_tensor(x_T):
        print(f"WARNING: Provide the input as sampled from resp timestep:{T}, as stochasticity is turned off. (ignore if not for evaluation)")

    model.eval()
    B = x_t.shape[0]

    for t in reversed(range(0, T_steps)):
        # expanded vector of 't' value accross entire batch (B)
        t_tensor = torch.full((B,), t, dtype=torch.long, device=device)
        # model predicts x0_hat (the model was trained to predict m0 directly)
        x0_hat = model(x_t, t_tensor, p2d)

        # compute parameters of q(x_{t-1} | x_t, x0_hat)
        if t > 0:
            # computation of eps_hat from x_t and x0_hat is not necessary
            a_bar_t = alphas_cumprod[t].to(device)
            a_bar_prev = alphas_cumprod[t-1].to(device)
            
            # Posterior Mean formula (eq from DPPM)
            coef1 = torch.sqrt(a_bar_prev) * (betas[t] / (1.0 - a_bar_t))
            coef1 = coef1.unsqueeze(0).unsqueeze(1).unsqueeze(2) # scalar --> (1, 1, 1)
            
            coef2 = torch.sqrt(alphas[t]) * (1.0 - a_bar_prev) / (1.0 - a_bar_t)
            coef2 = coef2.unsqueeze(0).unsqueeze(1).unsqueeze(2) # scalar --> (1, 1, 1)
            
            mu = coef1 * x0_hat + coef2 * x_t  # broadcast to (B,S,D)
            
            if stochasticity:
                # Posterior Variance formula (eq from DDPM)
                var = betas[t] * (1.0 - a_bar_prev) / (1.0 - a_bar_t)
                sigma = torch.sqrt(var)
                sigma = sigma.unsqueeze(0).unsqueeze(1).unsqueeze(2) # scalar --> (1, 1, 1)
    
                # Sample noise
                noise = torch.randn_like(x_t)
    
                # Stochastic update
                x_t = mu + sigma * noise  # <-- This is the DDPM formula!
            
            else:
                # Deterministic update
                x_t = mu  # do not add noise; use mean
        
        else:
            x_t = x0_hat # t == 0: final x0_hat is our best estimate

    model.train()
    return x_t

''' test case:

model.verbose = False
# test sequential denoise: sample x_T from a groundtruth x0
it = iter(dl) # reset
batch = next(it)
x0 = batch['m3d'].to(device)
p2d = batch['p2d'].to(device)
# create x_T by forward-noising to t=T-1 (largest noise)
t_Tminus1 = torch.full((x0.shape[0],), T-1, dtype=torch.long, device=device)
x_T, _ = q_sample(m3d_gt, t_Tminus1)
x0_hat = sequential_denoise(model, x_T, p2d)
print("sequential denoise finished; x0_hat shape:", x0_hat.shape)
'''