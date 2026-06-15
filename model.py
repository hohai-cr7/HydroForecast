import torch
import torch.nn as nn

import math, torch

def timestep_embedding(t: torch.Tensor, dim: int, max_period: int = 10_000):
    """
    t: [B] int/float -> [B, dim]
    """
    device = t.device
    half = dim // 2
    freqs = torch.exp(-math.log(max_period) * torch.arange(0, half, dtype=torch.float32, device=device) / half)
    args = t.float().unsqueeze(1) * freqs.unsqueeze(0)
    emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2 == 1:
        emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
    return emb

class CondDiffusion(nn.Module):
    
    def __init__(
        self,
        input_dim_fast:   int,   
        input_dim_future: int,   
        geo_dim:          int,   
        hidden_dim:       int = 128,
        num_layers:       int = 1,
        dropout:          float = 0.1,
        static_emb_dim:   int = 64,  
        use_layernorm_in: bool = True,
        pro=0.15
    ):
        super().__init__()
        H = hidden_dim
        self.H = H
        self.static_emb_dim = static_emb_dim
        self.use_ln = use_layernorm_in
        self.p_uncond = pro
        
        # ---- static embedding ----
        self.geo_norm = nn.LayerNorm(geo_dim) if use_layernorm_in else nn.Identity()
        self.geo_mlp  = nn.Sequential(
            nn.Linear(geo_dim, static_emb_dim),
            nn.SiLU(),
            nn.Linear(static_emb_dim, static_emb_dim)
        )

        # ---- input norms ----
        self.fast_norm   = nn.LayerNorm(input_dim_fast)   if use_layernorm_in else nn.Identity()
        self.future_norm = nn.LayerNorm(input_dim_future) if use_layernorm_in else nn.Identity()
       
        
        self.hindcast = nn.LSTM(input_dim_fast + static_emb_dim, H, num_layers=num_layers,
                                batch_first=True, dropout=dropout if num_layers > 1 else 0.0)

        
        self.future_encoder = nn.LSTM(input_dim_future, H, num_layers=num_layers,
                                      batch_first=True, dropout=dropout if num_layers > 1 else 0.0)

        
        self.t_emb_mlp = nn.Sequential(nn.Linear(H, H), nn.SiLU(), nn.Linear(H, H))
        
        
        cond_init_dim = H + H + static_emb_dim + H
        self.h_init = nn.Sequential(nn.Linear(cond_init_dim, H), nn.SiLU(), nn.Linear(H, H))
        c_init_dim = H + H
        self.c_init = nn.Sequential(nn.Linear(c_init_dim, H), nn.SiLU(), nn.Linear(H, H))

        
        self.runoff_encoder = nn.Linear(1, H) 

        
        dec_in_dim = H + static_emb_dim + H 
        self.decoder = nn.LSTM(dec_in_dim, H, num_layers=num_layers,
                               batch_first=True, dropout=dropout if num_layers > 1 else 0.0)

        self.dropout = nn.Dropout(dropout)
        self.head    = nn.Linear(H, 1)

        
        self._init_lstm_forget_bias(self.hindcast)
        self._init_lstm_forget_bias(self.future_encoder)
        self._init_lstm_forget_bias(self.decoder)

    @staticmethod
    def _init_lstm_forget_bias(lstm: nn.LSTM, forget_bias: float = 1.0):
        
        for names in lstm._all_weights:
            for name in filter(lambda n: "bias" in n, names):
                b = getattr(lstm, name)
                nn.init.zeros_(b)
                hidden = b.shape[0] // 4
                b.data[hidden:2*hidden] = forget_bias

    def forward(
        self,
        noisy_runoff: torch.Tensor,   # [B, T_out, 1]
        t: torch.Tensor,              # [B]
        past_fast: torch.Tensor,      # [B, L_past, F_fast]
        future_weather: torch.Tensor, # [B, T_out, F_future]
        geo_data: torch.Tensor,       # [B, G]
        force_uncond: bool = False,   
        force_drop_future = False
        
    ):
        B, T_out, _ = noisy_runoff.shape
        H = self.H
        device = future_weather.device

        # ---- Classifier-Free Guidance (CFG) Dropout ----
        if force_uncond:
            drop = torch.ones((B, 1, 1), device=device, dtype=future_weather.dtype)
        else:
            if self.training and self.p_uncond > 0:
                bern = (torch.rand(B, device=device) < self.p_uncond).to(future_weather.dtype)   # zero represent mean weather forcing (globel mean, maybe station mean is better) 
                drop = bern.view(B, 1, 1)
            else:
                drop = torch.zeros((B, 1, 1), device=device, dtype=future_weather.dtype)  
        
       
        future_weather = future_weather * (1.0 - drop)
        
        
        
        # Static 
        g = self.geo_norm(geo_data)
        s_emb = self.geo_mlp(g)                                     # [B, E]
        s_past = s_emb.unsqueeze(1).expand(-1, past_fast.size(1), -1)    # [B, L_past, E]
        s_fut  = s_emb.unsqueeze(1).expand(-1, T_out, -1)           # [B, T_out, E]
        
        # Past 
        pf = self.fast_norm(past_fast)
        hind_in = torch.cat([pf, s_past], dim=-1)                   # [B, L_past, F_fast+E]
        _, (h_past, c_past) = self.hindcast(hind_in)                # [num_layers, B, H]
        h_past = h_past[-1]                                         # [B, H] 
        c_past = c_past[-1]                                         # [B, H]
        
       
        fut = self.future_norm(future_weather)
        _, (h_future, c_future) = self.future_encoder(fut)          # [num_layers, B, H]
        h_future = h_future[-1]                                     # [B, H]
        c_future = c_future[-1]                                     # [B, H]
        
        # Time 
        t_emb = timestep_embedding(t, H)                            # [B, H]
        t_e   = self.t_emb_mlp(t_emb)
        t_broadcast = t_e.unsqueeze(1).expand(-1, T_out, -1)        # [B, T_out, H]
        
        
       
        cond_init_vec = torch.cat([h_past, h_future, t_e, s_emb], dim=-1)
        h0 = self.h_init(cond_init_vec)                             # [B, H]
        c_init_vec = torch.cat([c_past, c_future], dim=-1)   # not add the future weather to decoder in, maybe this better
        
        c0 = self.c_init(c_init_vec)                             # [B, H]
        h0 = h0.unsqueeze(0) # [1, B, H] 
        c0 = c0.unsqueeze(0) # [1, B, H]
        
        # ---- 4. Decoder  ----
        
        
        runoff_emb = self.runoff_encoder(noisy_runoff)              # [B, T_out, H]

        
        dec_in = torch.cat([runoff_emb, s_fut, t_broadcast], dim=-1) # [B, T_out, H+E+H]
        
        
        dec_out, _ = self.decoder(dec_in, (h0, c0))                 # [B, T_out, H]
        
        # ---- Output ----
        dec_out = self.dropout(dec_out)
        eps = self.head(dec_out)                                    # [B, T_out, 1]
        
        return eps
    

