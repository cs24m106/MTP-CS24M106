import math, torch
import torch.nn as nn

class PositionalEncoding(nn.Module):
    """
    Common Position Encoder model (fixed, i.e. non-learnable) 
    for sequences ([B, 0...S, d] -> [B, 0..S, d]) & timesteps ([B,1]->[B,d])
    Parameters:
        - `d_model` : feature dimension (embedding size).
        - `max_len` : maximum sequence length supported.

    Forward:
        - input `x`: (Batch, S-Frames, C-HiddenLayerChannels)
        - HiddenLayer => projected feature dimention of 2d-pose/3d-motion
        - returns: x + PE
    """
    def __init__(self, d_model, max_len=1000):
        super().__init__()
        self.d_model = d_model
        self.max_len = max_len
        pe = torch.zeros(max_len, d_model)

        # arrange => [0..max_len-1], dim=1d-vector (i.e. dim idx=0), unsqueeze(1) => add dim at idx=1. 
        pos = torch.arange(0, max_len).unsqueeze(1).float()
        # so pos.shape = (max_len, 1), i.e. its a column vector
        
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)) # shape=(d_model/2)
        # denominator (frequency) => 10000^(2*i/d_model), where i is the embedding index
        # pos*div_term => [max_len,1]*[1,d_model], i.e. broadcasting to (max_len, d_model/2)
        
        pe[:, 0::2] = torch.sin(pos * div_term) # Fills even cols (0, 2, 4, ...) of embeddings with sine(pos*freq)
        pe[:, 1::2] = torch.cos(pos * div_term) # Fills odd cols (1, 3, 5, ...) of embeddings with cos(pos*freq)
        self.register_buffer('pe', pe) # Registers as a model.buffer (not a parameter, for faster access)

    def sequential_forward(self, x): # Forward for sequence data: x shape (B,S,C)
        assert x.size(-1) == self.pe.size(1), f"PE_seq: Feature Dimension (d_model:{self.d_model}) mismatch!"
        S = x.size(1)
        assert S <= self.pe.size(0), f"PE_seq: Length of Sequence exceeds max_len({self.max_len}) contraint, increase the param val!"
        return x + self.pe[:S].unsqueeze(0) # shapes: x=(B,S,C) + pe=(1,S,C) -> (B,S,c) how?
        # Adding (B,S,C) + (1,S,C) broadcasts the leading 1 to B, producing (B,S,C). So you do NOT need to manually repeat pe per batch

    def timestep_forward(self, t): # Forward for timesteps: t shape (B,)
        return self.pe[t] # returns (B,d_model)

    def forward(self, x):
        assert x.dim() <= 3, "Position Encoder cant handle more than 3 dimentions, reshape accordingly!"
        if (x.dim() == 3):  # sequence data
            return self.sequential_forward(x)
        else: # timestep data
            if (x.dim() == 2) and (x.size(-1) == 1):
                x = x.squeeze(1)  # make it (B,)  1D list for indexing
            if x.dim() == 1:
                return self.timestep_forward(x)
            else:
                raise AssertionError(f"Position Encoder not implemented for {x.dim()} dimention inputs!")


# ======================== ViMo-Denoiser FrameWork ========================

class DenoiserBlock(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        # fundamental blocks of denoiser
        self.self_attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=n_heads, batch_first=True) # self-attention for motion tokens
        self.cross_attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=n_heads, batch_first=True) # cross-attention with pose tokens
        self.ff_mlp = nn.Sequential(nn.Linear(d_model, d_model*4), nn.GELU(), nn.Linear(d_model*4, d_model)) # feed-forward MLPs with GELU act-fn
        # to preserve the pipeline order via adding residual + norm around each major sub-block (self-attn, cross-attn, MLP)
        self.norm_sa = nn.LayerNorm(d_model) # layer norm after self-attn
        self.norm_ca = nn.LayerNorm(d_model) # layer norm after cross-attn
        self.norm_ff = nn.LayerNorm(d_model) # layer norm after ff-mlp

    def forward(self, motion_tokens, pose_tokens, gamma, beta): # film params remains same within single block?
        # because gamma & beta are generated from conditions => encoded(pose), embed(timestep) which are constant per denoising step
        x = motion_tokens # alias overwriting after each block

        # Self-attention (Q:m3d, K:m3d, V:m3d) -> learns its own temporal dependencies
        sa = self.self_attn(query=x, key=x, value=x, need_weights=False)[0]
        x = self.norm_sa(x + sa)
        x = x * (1 + gamma) + beta # apply FiLM after self-attn

        # Cross-attention (Q:m3d, K:p2d, V:p2d) -> conditioned guidance on pose tokens
        ca = self.cross_attn(query=x, key=pose_tokens, value=pose_tokens, need_weights=False)[0]
        x = self.norm_ca(x + ca)
        x = x * (1 + gamma) + beta # apply FiLM after self-attn
        
        # Feed-Forward MLP (finally to introduce non-linearity)
        fm = self.ff_mlp(x)
        x = self.norm_ff(x + fm)
        x = x * (1 + gamma) + beta # apply FiLM after MLP
        return x


class ViMoFrameWork(nn.Module):
    """
    Parameters:
        - motion_dim: flattened dimension of 3D motion input (default 24*6+4+3=151)
        - pose_dim: flattened dimension of 2D pose input (default 17*3=51)
        - max_T: maximum diffusion timesteps (default 1000)
        - embed_dim: Transformer embedding dimension (default 256)
        - n_heads: no. of Attention Heads (parallel subspaces) in Transformer Layers (default 8)
        - m_nlayers: no. of Denoiser Blocks in 3D motion ViMo Framework (default 3)
        - p_nlayers: no. of Transformer Layers in 2D pose encoder (default 2)
        - cond_drop: classifier-free guidance drop probability during training (default 0.25 => 25% unconditioned, 75% conditioned)
        
    Forward:
        - input `x_t`: (B, S, motion_dim) - noisy 3d motion
        - input `timestep`: (B,) long or float on device in [0..T-1]
        - (optional) `p2d`: (B, S, 17, 3) or (B, S, pose_dim) - 2d poses (None for unconditioned)
        - (optional) `drop_off`: probability to drop off the guidance (poses conditioned training)
          unconditioned rate (overrides self.cond_drop if provided), this option for eval-time experiments
        - returns: predicted noise (B, S, motion_dim)
    """
    def __init__(self, motion_dim=151, pose_dim=17*3, max_T=1000, embed_dim=256, n_heads=8, m_nlayers=3, p_nlayers=2, cond_drop=0.25, verbose=False):
        super().__init__()
        assert embed_dim % n_heads == 0, "Embedding dimension must be divisible by the number of heads."
        # save all config for easier access through vars(model) or model.__dict__
        self.motion_dim = motion_dim
        self.pose_dim = pose_dim
        self.max_T = max_T
        self.embed_dim = embed_dim
        self.n_heads = n_heads
        self.m_nlayers = m_nlayers
        self.p_nlayers = p_nlayers
        self.cond_drop = cond_drop
        self.verbose = verbose
        self.PE = PositionalEncoding(embed_dim, 2*max(motion_dim, pose_dim, max_T)) # shared PE instance

        # motion projections
        self.m3d_features = nn.Linear(motion_dim, embed_dim)

        # pose encoder (temporal aggregator) project per-frame pose => tokens
        self.p2d_features = nn.Linear(pose_dim, embed_dim)  # map flattened per-frame pose to embed_dim
        encoder_layer_p = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=n_heads, dim_feedforward=embed_dim*4, batch_first=True) # (B,S,C)
        self.pose_encoder = nn.TransformerEncoder(encoder_layer_p, num_layers=p_nlayers) # default: num_layers=2 (for the pose encoder) why?
        # pose seq are relatively low-dim, so a small encoder captures temporal dependencies while keeping computation lesser.

        # diffusion embed(denoiser_step T)-> use small MLP to produce FiLM params
        self.timestep_embedding = nn.Sequential(nn.Linear(embed_dim, embed_dim), nn.ReLU(), nn.Linear(embed_dim, embed_dim))
        
        # stack of denoising modules (each contains self-attn, cross-attn, MLP)
        self.d_modules = nn.ModuleList([DenoiserBlock(embed_dim, n_heads) for _ in range(m_nlayers)])

        # layers to produce FiLM parameters from concatenated[encoded(pose); embeded(timestep)]
        self.film_mlp = nn.Linear(embed_dim*2, embed_dim*2)  # -> gamma/beta flattened
        nn.init.zeros_(self.film_mlp.weight); nn.init.zeros_(self.film_mlp.bias) # to preserve identify intial training

        # output projection back to motion space
        self.out_proj = nn.Linear(embed_dim, motion_dim)

    def forward(self, x_t, timestep, p2d=None, drop_off=None):
        B, S, _ = x_t.shape
        cond_drop = drop_off if drop_off is not None else self.cond_drop
        if self.verbose: print(f"x_t.shape: {x_t.shape}, timestep.shape: {timestep.shape}, p2d.shape: {p2d.shape if p2d is not None else "None"} ...")
        if timestep.dim() == 1:
            timestep = timestep.unsqueeze(-1)  # (B,) -> (B,1) our common PE expects in this form

        # apply classifier-free drop to condition during training
        if p2d is None or cond_drop == 1:
            p_flat = torch.zeros(B, S, self.pose_dim, device=x_t.device)
        else:
            p_flat = p2d.view(B, S, -1) if p2d.dim() > 3 else p2d # flatten per-frame pose if needed
            drop_mask = (torch.rand(B, device=x_t.device) < cond_drop).int().view(B,1,1) # generate random for each batch
            p_flat = p_flat * (1 - drop_mask) # zero-out where dropped, (B,S,pose_dim)*(B,1,1) broadcasts over dim-1,2
            if self.verbose: print(f"drop_mask.shape: {drop_mask.shape},", end=" ")
        if self.verbose: print(f"p_flat.shape: {p_flat.shape},", end=" ")
        
        assert not (torch.isnan(p_flat).any() or torch.isnan(x_t).any() or torch.isnan(timestep).any()), "Model Input should not have nan values!"
        # project pose tokens and encode temporally
        p_embed_raw = self.p2d_features(p_flat)  # (B,S,embed_dim)
        if self.verbose: print(f"p_embed_raw.shape: {p_embed_raw.shape},", end=" ")
        p_embed_pos = self.PE(p_embed_raw)
        if self.verbose: print(f"p_embed_pos.shape: {p_embed_pos.shape},", end=" ")
        p_encoded = self.pose_encoder(p_embed_pos)  # (B, S, d) here batch_first=True
        if self.verbose: print(f"p_encoded.shape: {p_encoded.shape} ...")

        # timestep embedding
        t_pos = self.PE(timestep)  # sinusoidal positional embeddings (B,1) -> (B,embed_dim)
        if self.verbose: print(f"t_pos.shape: {t_pos.shape},", end=" ")
        t_emb = self.timestep_embedding(t_pos)  # (B, embed_dim)
        if self.verbose: print(f"t_emb.shape: {t_emb.shape} ...")
        
        # Add time position encodings to pose tokens
        pt_emb = p_encoded + t_pos.unsqueeze(1)  # (B,S,d) + (B,1,d) => broadcast across frame axis
        if self.verbose: print(f"pt_emb.shape: {pt_emb.shape},", end=" ")
        # Mean pooling: aggregate pose into a single global cond vec representing the whole seq
        p_avg = pt_emb.mean(dim=1)  # (B, sum(embed_dim)/S_frames) i.e. mean over time
        if self.verbose: print(f"p_avg.shape: {p_avg.shape} ...")

        # combine encoded(pose), embedding(timestep) -> FiLM params
        film_in = torch.cat([p_avg, t_emb], dim=-1) # concat on last dim
        if self.verbose: print(f"film_in.shape: {film_in.shape},", end=" ")
        film_params = self.film_mlp(film_in)  # (B, 2*embed_dim)
        if self.verbose: print(f"film_params.shape: {film_params.shape},", end=" ")
        gamma, beta = film_params.chunk(2, dim=-1)  # split 2 chunks across last dim, so each (B, embed_dim)
        # expand (B,1,d) to (B,S,d)
        gamma = gamma.unsqueeze(1).expand(-1, S, -1)
        beta  = beta.unsqueeze(1).expand(-1, S, -1)
        if self.verbose: print(f"gamma.shape: {gamma.shape}, beta.shape: {beta.shape} ...")

        # motion projection + pos enc
        x = self.m3d_features(x_t)  # (B,S,d)
        if self.verbose: print(f"m3d_features.shape: {x.shape},", end=" ")
        x = self.PE(x) 
        if self.verbose: print(f"m3d_PE.shape: {x.shape},", end=" ")
        for denoiser in self.d_modules:
            x = denoiser(x, pt_emb, gamma, beta)  # (S,B,d)
            if self.verbose: print(f"m3d_denoiser.shape: {x.shape},", end=" ")
        
        # finally project back to motion_dim
        out = self.out_proj(x)
        if self.verbose: print(f"m3d_out.shape: {out.shape} ...\n")
        return out
    
    @property # adding device property to easier access
    def device(self):
        return next(self.parameters()).device