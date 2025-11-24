import torch

# in-scipt copy to avoid pass-in param for sampling
curr_alpha_bar = None 
curr_betas = None
curr_alphas = None
T = None

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
    global curr_alphas
    curr_alphas = alphas
    alphas_cumprod = torch.cumprod(alphas, dim=0)  # \bar{alpha}_t => cummulative product
    global curr_alpha_bar
    curr_alpha_bar = alphas_cumprod

# ------------------ In-Module Helper Utility ------------------
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
    global T
    T = steps
    return betas

# ------------------ Training Helper Utility ------------------
def forward_diffusion_sample(x0, t, betas=None):
    """
    Forward Noising: q_sample: x_t = sqrt(alpha_bar) * x0 + sqrt(1-alpha_bar) * noise
    Input Parameters:
        x0: ground truth input at t=0, here motion data (B, S, D)
        t: (B,) different timesteps for each item in batch, (B,S) PTSS, long tensor in [0..T-1]
        betas: noise per timestep tensor (must, if make_beta_schedule() wasn't previously called)
    """
    check_betas(betas)
    alphas_cumprod = curr_alpha_bar
    
    noise = torch.randn_like(x0) # same shape as x0 or None (then sampled)
    a_bar_batch = alphas_cumprod[t] # gather alpha_bar for each batch/frame

    if t.dim() == 1:    
        a_bar = a_bar_batch.view(-1, 1, 1).to(x0.device)  # (B,1,1)
    else: # PTSS dim=2, (B, S)
        a_bar = a_bar_batch.unsqueeze(-1).to(x0.device)  # (B, S, 1)
    #-1 means "infer this dimension from the length of the input".
    # The two 1s add singleton dimensions, making it easy to broadcast in later operations.

    xt = torch.sqrt(a_bar) * x0 + torch.sqrt(1.0 - a_bar) * noise
    return xt

# ------------------ Training Helper Utility ------------------
def probabilistic_timestep_sampling(B, S, T, p=0.2): # .2 is best value according to refered paper
    """
    Probabilistic Timestep Sampling Strategy (PTSS)
    With probability p: sample independent timesteps per frame
    With probability (1-p): use same timestep for all frames
    Vectorized Timestep Approach: https://arxiv.org/html/2410.03160v1
    """
    timesteps = torch.zeros(B, S, dtype=torch.long)
    for i in range(B):
        if torch.rand(1).item() < p:
            timesteps[i] = torch.randint(0, T, (S,))  # Independent
        else:
            t = torch.randint(0, T, (1,))
            timesteps[i] = t.repeat(S)  # Same for all frames
    return timesteps

# ------------------ Loss Helper Utility ------------------
def get_snr_weight(timestep, type="cp", w_cap=1.0, batch_norm=False, ratio=0.05):
    """
    Flexible SNR-based loss weighting for diffusion training
    
    Parameters:
        timestep: (B,)/(B,S) batchwise timestep indices
        type: String specifying weighting scheme (see below)
        w_cap: Weight cap/maximum (default: 1.0)
        batch_norm: Whether to normalize by batch mean (default: False)
        ratio: Blending ratio for averaging methods (default: 0.05)
    
    Type specifications:
        Min-cap methods (single component):
            '' or 'none'  : No SNR weighting (all ones)
            'cp'          : min(SNR, cap)
            'sq'          : min(SNR, sqrt_decay)
            'lr'          : min(SNR, linear_decay)
        
        Averaging methods (two components separated by '/'):
            Format: '<left>/<right>' means avg(left_weights, right_decay)
            
            Left component (SNR-based):
                'cp' : min(SNR, cap)
                'sq' : min(SNR, sqrt_decay)
                'lr' : min(SNR, linear_decay)
            
            Right component (decay benchmark):
                ''   : no_decay (constant cap)
                'sq' : sqrt_decay
                'lr' : linear_decay
            
            Examples:
                'cp/'    : avg(min(SNR,cap), no_decay)
                'sq/sq'  : avg(min(SNR,sqrt), sqrt_decay)  [RECOMMENDED]
                'lr/lr'  : avg(min(SNR,linear), linear_decay)
    
    Returns:
        weight: (B,) tensor of loss weights
    """
    # Get alpha_bar for timestep
    alpha_bar_t = curr_alpha_bar[timestep]  # (B,)
    snr = alpha_bar_t / (1.0 - alpha_bar_t + 1e-8)
    
    # Prepare decay schedules
    t_norm = timestep.float() / T
    no_decay = w_cap * torch.ones_like(t_norm)
    sqrt_decay = w_cap * torch.sqrt(1.0 - t_norm)
    linear_decay = w_cap * (1.0 - t_norm)
    
    # Parse type string
    type = type.lower().strip()
    
    # No SNR weighting
    if type == '' or type == 'none':
        weight = no_decay
    
    # Min-cap methods (single component)
    elif type == 'cp':
        weight = torch.minimum(snr, no_decay)
    
    elif type == 'sq':
        weight = torch.minimum(snr, sqrt_decay)
    
    elif type == 'lr':
        weight = torch.minimum(snr, linear_decay)
    
    # Averaging methods (two components)
    elif '/' in type:
        parts = type.split('/')
        if len(parts) != 2:
            raise ValueError(f"Invalid type format: '{type}'. Use '<left>/<right>'")
        
        left_type, right_type = parts[0].strip(), parts[1].strip()
        
        # Compute left component (SNR-based weights)
        if left_type == 'cp':
            left_weights = torch.minimum(snr, no_decay)
        elif left_type == 'sq':
            left_weights = torch.minimum(snr, sqrt_decay)
        elif left_type == 'lr':
            left_weights = torch.minimum(snr, linear_decay)
        else:
            raise ValueError(f"Invalid left component: '{left_type}'. Use 'cp', 'sq', or 'lr'")
        
        # Compute right component (decay benchmark)
        if right_type == '' or right_type == 'none':
            right_decay = no_decay
        elif right_type == 'sq':
            right_decay = sqrt_decay
        elif right_type == 'lr':
            right_decay = linear_decay
        else:
            raise ValueError(f"Invalid right component: '{right_type}'. Use '', 'sq', or 'lr'")
        
        # Average with ratio
        weight = (1 - ratio) * left_weights + ratio * right_decay
    
    else:
        raise ValueError(f"Unknown type: '{type}'")
    
    # Optional batch normalization
    if batch_norm:
        weight = weight / (weight.mean() + 1e-8)
    return weight


# ------------------Evaluation Helper Utility ------------------
@torch.no_grad()
def reverse_denoising_sample(model, x_T=(1,150,151), T_steps=1000,  p2d=None, stochasticity=False, betas=None, verbose=False):
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
    alphas = curr_alphas
    alphas_cumprod = curr_alpha_bar
    
    device = model.device
    if isinstance(x_T, tuple):
        x_t = torch.randn(x_T, dtype=torch.float32, device=device) # randn = Gaussian noise, rand = Uniform noise
    elif torch.is_tensor(x_T):
        x_t = x_T.clone().to(device)
    else:
        raise RuntimeError(f"Given format for input x_T is unrecognized. Type: {type(x_T)}")
    
    if stochasticity and not torch.is_tensor(x_T):
        print(f"WARNING: Provide the input as sampled from resp timestep:{T_steps}, as stochasticity is turned off. (ignore if not for evaluation)")

    model.eval()
    B = x_t.shape[0]

    for t in reversed(range(0, T_steps)):
        # expanded vector of 't' value accross entire batch (B)
        t_tensor = torch.full((B,), t, dtype=torch.long, device=device)
        # model predicts x0_hat (the model was trained to predict m0 directly)
        if verbose and t==T_steps-1: model.verbose = True; print("Single Model debugging:")
        x0_hat = model(x_t, t_tensor, p2d, drop_off=0) # only conditioned --> end goal
        if verbose and t==T_steps-1: model.verbose = False
        
        # compute parameters of q(x_{t-1} | x_t, x0_hat)
        if t > 0:
            # computation of eps_hat from x_t and x0_hat is not necessary
            a_bar_t = alphas_cumprod[t].to(device)
            a_bar_prev = alphas_cumprod[t-1].to(device)
            
            # Posterior Mean formula (eq from DPPM)
            coef1 = torch.sqrt(a_bar_prev) * (betas[t] / (1.0 - a_bar_t))
            coef2 = torch.sqrt(alphas[t]) * (1.0 - a_bar_prev) / (1.0 - a_bar_t)
            if verbose and not stochasticity: print(f"> Posterior formula [T={t}] : Mean only => coef1 = {coef1.item():.4f}, coef2 = {coef2.item():.4f}")

            coef1 = coef1.unsqueeze(0).unsqueeze(1).unsqueeze(2) # scalar --> (1, 1, 1)
            coef2 = coef2.unsqueeze(0).unsqueeze(1).unsqueeze(2)
            mu = coef1 * x0_hat + coef2 * x_t  # broadcast to (B,S,D)
            
            if stochasticity:
                # Posterior Variance formula (eq from DDPM)
                var = betas[t] * (1.0 - a_bar_prev) / (1.0 - a_bar_t)
                if verbose: print(f"> Posterior formula [T={t}] : Mean & Variance => miu_coef1 = {coef1.item():.4f}, miu_coef2 = {coef2.item():.4f}; var = {var.item():.4f}")
                
                sigma = torch.sqrt(var)
                sigma = sigma.unsqueeze(0).unsqueeze(1).unsqueeze(2) # scalar --> (1, 1, 1)
                noise = torch.randn_like(x_t) # Sample noise
    
                # Stochastic update
                x_t = mu + sigma * noise  # <-- This is the DDPM formula!
            
            else:
                # Deterministic update
                x_t = mu  # do not add noise; use mean
        
        else:
            x_t = x0_hat # t == 0: final x0_hat is our best estimate

    model.train()
    return x_t
