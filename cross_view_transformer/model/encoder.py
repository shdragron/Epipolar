import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange, repeat
from torchvision.models.resnet import Bottleneck
from typing import List, Optional

try:
    import wandb
    WANDB_AVAILABLE = True  # Set to True to enable custom EAF logging
except ImportError:
    WANDB_AVAILABLE = False


ResNetBottleNeck = lambda c: Bottleneck(c, c // 4)


def generate_grid(height: int, width: int):
    xs = torch.linspace(0, 1, width)
    ys = torch.linspace(0, 1, height)

    # Create meshgrid - compatible with older PyTorch versions
    grid_y, grid_x = torch.meshgrid(ys, xs)
    indices = torch.stack([grid_x, grid_y], 0)                            # 2 h w
    indices = F.pad(indices, (0, 0, 0, 0, 0, 1), value=1)                   # 3 h w
    indices = indices[None]                                                 # 1 3 h w

    return indices


def get_view_matrix(h=200, w=200, h_meters=100.0, w_meters=100.0, offset=0.0):
    """
    copied from ..data.common but want to keep models standalone
    """
    sh = h / h_meters
    sw = w / w_meters

    return [
        [ 0., -sw,          w/2.],
        [-sh,  0., h*offset+h/2.],
        [ 0.,  0.,            1.]
    ]


class Normalize(nn.Module):
    def __init__(self, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
        super().__init__()

        self.register_buffer('mean', torch.tensor(mean)[None, :, None, None], persistent=False)
        self.register_buffer('std', torch.tensor(std)[None, :, None, None], persistent=False)

    def forward(self, x):
        return (x - self.mean) / self.std


class RandomCos(nn.Module):
    def __init__(self, *args, stride=1, padding=0, **kwargs):
        super().__init__()

        linear = nn.Conv2d(*args, **kwargs)

        self.register_buffer('weight', linear.weight)
        self.register_buffer('bias', linear.bias)
        self.kwargs = {
            'stride': stride,
            'padding': padding,
        }

    def forward(self, x):
        return torch.cos(F.conv2d(x, self.weight, self.bias, **self.kwargs))


class BEVEmbedding(nn.Module):
    def __init__(
        self,
        dim: int,
        sigma: int,
        bev_height: int,
        bev_width: int,
        h_meters: int,
        w_meters: int,
        offset: int,
        decoder_blocks: list,
    ):
        """
        Only real arguments are:

        dim: embedding size
        sigma: scale for initializing embedding

        The rest of the arguments are used for constructing the view matrix.

        In hindsight we should have just specified the view matrix in config
        and passed in the view matrix...
        """
        super().__init__()

        # each decoder block upsamples the bev embedding by a factor of 2
        h = bev_height // (2 ** len(decoder_blocks))
        w = bev_width // (2 ** len(decoder_blocks))

        # bev coordinates
        grid = generate_grid(h, w).squeeze(0)
        grid[0] = bev_width * grid[0]
        grid[1] = bev_height * grid[1]

        # map from bev coordinates to ego frame
        V = torch.tensor(get_view_matrix(bev_height, bev_width, h_meters, w_meters, offset),
                        dtype=grid.dtype, device=grid.device)
        V_inv = torch.linalg.inv(V)                                 # 3 3
        grid = V_inv @ rearrange(grid, 'd h w -> d (h w)')                      # 3 (h w)
        grid = rearrange(grid, 'd (h w) -> d h w', h=h, w=w)                    # 3 h w

        # egocentric frame
        self.register_buffer('grid', grid, persistent=False)                    # 3 h w
        self.learned_features = nn.Parameter(sigma * torch.randn(dim, h, w))    # d h w

    def get_prior(self):
        return self.learned_features


class CrossAttentionEAF(nn.Module):
    """
    Cross-view attention with Epipolar Attention Field (EAF).

    입력:
      q: [B, D, H_bev, W_bev]          # BEV features
      k: [B, N, D, h, w]               # per-view image features (after proj)
      v: [B, N, D, h, w]
      W_logits: [B, Q, NK]             # EAF weights (Q = H_bev*W_bev, NK = N*h*w)
      skip: [B, D, H_bev, W_bev] or None

    출력:
      z: [B, D, H_bev, W_bev]
    """
    def __init__(self, dim, heads, dim_head, qkv_bias, norm=nn.LayerNorm):
        super().__init__()
        self.heads = heads
        self.dim_head = dim_head
        self.scale = dim_head ** -0.5

        inner_dim = heads * dim_head

        self.q_norm = norm(dim)
        self.k_norm = norm(dim)
        self.v_norm = norm(dim)

        self.to_q = nn.Linear(dim, inner_dim, bias=qkv_bias)
        self.to_k = nn.Linear(dim, inner_dim, bias=qkv_bias)
        self.to_v = nn.Linear(dim, inner_dim, bias=qkv_bias)

        self.proj = nn.Linear(inner_dim, dim)

        self.prenorm = norm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, 2 * dim),
            nn.GELU(),
            nn.Linear(2 * dim, dim),
        )
        self.postnorm = norm(dim)

    def forward(self, q, k, v, W_logits, vis_flat, skip=None):
        """
        q: [B, D, H_bev, W_bev]
        k: [B, N, D, h, w]
        v: [B, N, D, h, w]
        W_logits: [B, Q, NK]
        vis_flat: [B, Q, NK]  # visibility mask (0 or 1)
        """
        B, D, H_bev, W_bev = q.shape
        Bk, N, Dk, h, w = k.shape
        assert B == Bk and D == Dk, "q/k shape mismatch"

        Q = H_bev * W_bev
        K_per = h * w
        NK = N * K_per

        # ---- reshape ----
        # q: [B, Q, D]
        q = rearrange(q, 'b d H W -> b (H W) d')

        # k,v: [B, NK, D]
        k = rearrange(k, 'b n d h w -> b (n h w) d')
        v = rearrange(v, 'b n d h w -> b (n h w) d')

        # ---- norm ----
        q = self.q_norm(q)
        k = self.k_norm(k)
        v = self.v_norm(v)

        # ---- projections ----
        q = self.to_q(q)   # [B, Q, H*Dh]
        k = self.to_k(k)   # [B, NK, H*Dh]
        v = self.to_v(v)   # [B, NK, H*Dh]

        # ---- split heads ----
        q = rearrange(q, 'b Q (h d) -> b h Q d', h=self.heads, d=self.dim_head)
        k = rearrange(k, 'b K (h d) -> b h K d', h=self.heads, d=self.dim_head)
        v = rearrange(v, 'b K (h d) -> b h K d', h=self.heads, d=self.dim_head)

        # ---- scaled dot product ----
        # logits: [B, H, Q, NK]
        logits = torch.einsum('b h Q d, b h K d -> b h Q K', q, k) * self.scale

        # ---- mask invisible keys ----
        # vis_flat: [B, Q, NK] -> [B, 1, Q, NK], broadcast to [B, H, Q, NK]
        # Use dtype-safe minimum value for fp16/fp32 compatibility
        neg_inf = torch.finfo(logits.dtype).min
        logits = logits.masked_fill(vis_flat.unsqueeze(1) == 0, neg_inf)

        # ---- apply Epipolar weights (multiplicative, as in W ⊙ (QK^T/√d)) ----
        # W_logits: [B, Q, NK] -> [B, 1, Q, NK], broadcast to [B, H, Q, NK]
        W = W_logits.unsqueeze(1)  # Broadcasting instead of expand
        logits = logits * W

        # ---- softmax over keys ----
        attn = torch.softmax(logits, dim=-1)   # [B, H, Q, NK]

        # ---- aggregate values ----
        out = torch.einsum('b h Q K, b h K d -> b h Q d', attn, v)  # [B,H,Q,Dh]
        out = rearrange(out, 'b h Q d -> b Q (h d)')                # [B,Q,inner]
        out = self.proj(out)                                       # [B,Q,D]

        # ---- reshape back to BEV map ----
        z = rearrange(out, 'b (H W) d -> b d H W', H=H_bev, W=W_bev)

        # ---- residual & FFN ----
        if skip is not None:
            z = z + skip

        z = self.prenorm(rearrange(z, 'b d H W -> b (H W) d'))
        z = z + self.mlp(z)
        z = self.postnorm(z)
        z = rearrange(z, 'b (H W) d -> b d H W', H=H_bev, W=W_bev)

        return z


class CrossAttention(nn.Module):
    def __init__(self, dim, heads, dim_head, qkv_bias, norm=nn.LayerNorm):
        super().__init__()

        self.scale = dim_head ** -0.5

        self.heads = heads
        self.dim_head = dim_head

        self.to_q = nn.Sequential(norm(dim), nn.Linear(dim, heads * dim_head, bias=qkv_bias))
        self.to_k = nn.Sequential(norm(dim), nn.Linear(dim, heads * dim_head, bias=qkv_bias))
        self.to_v = nn.Sequential(norm(dim), nn.Linear(dim, heads * dim_head, bias=qkv_bias))

        self.proj = nn.Linear(heads * dim_head, dim)
        self.prenorm = norm(dim)
        self.mlp = nn.Sequential(nn.Linear(dim, 2 * dim), nn.GELU(), nn.Linear(2 * dim, dim))
        self.postnorm = norm(dim)

    def forward(self, q, k, v, skip=None):
        """
        q: (b n d H W)
        k: (b n d h w)
        v: (b n d h w)
        """
        _, _, _, H, W = q.shape

        # Move feature dim to last for multi-head proj
        q = rearrange(q, 'b n d H W -> b n (H W) d')
        k = rearrange(k, 'b n d h w -> b n (h w) d')
        v = rearrange(v, 'b n d h w -> b (n h w) d')

        # Project with multiple heads
        q = self.to_q(q)                                # b (n H W) (heads dim_head)
        k = self.to_k(k)                                # b (n h w) (heads dim_head)
        v = self.to_v(v)                                # b (n h w) (heads dim_head)

        # Group the head dim with batch dim
        q = rearrange(q, 'b ... (m d) -> (b m) ... d', m=self.heads, d=self.dim_head)
        k = rearrange(k, 'b ... (m d) -> (b m) ... d', m=self.heads, d=self.dim_head)
        v = rearrange(v, 'b ... (m d) -> (b m) ... d', m=self.heads, d=self.dim_head)

        # Dot product attention along cameras
        dot = self.scale * torch.einsum('b n Q d, b n K d -> b n Q K', q, k)
        dot = rearrange(dot, 'b n Q K -> b Q (n K)')
        att = dot.softmax(dim=-1)

        # Combine values (image level features).
        a = torch.einsum('b Q K, b K d -> b Q d', att, v)
        a = rearrange(a, '(b m) ... d -> b ... (m d)', m=self.heads, d=self.dim_head)

        # Combine multiple heads
        z = self.proj(a)

        # Optional skip connection
        if skip is not None:
            z = z + rearrange(skip, 'b d H W -> b (H W) d')

        z = self.prenorm(z)
        z = z + self.mlp(z)
        z = self.postnorm(z)
        z = rearrange(z, 'b (H W) d -> b d H W', H=H, W=W)

        return z


class CrossViewAttention(nn.Module):
    def __init__(
        self,
        feat_height: int,
        feat_width: int,
        feat_dim: int,
        dim: int,
        image_height: int,
        image_width: int,
        qkv_bias: bool,
        heads: int = 4,
        dim_head: int = 32,
        no_image_features: bool = False,
        skip: bool = True,
        eaf_lambda: float = 1.0,
        eaf_zmax: float = 4.0,
        use_adaptive_lambda: bool = True,
        min_sigma: float = 1.0,
        max_sigma: float = 8.0,
    ):
        super().__init__()

        # 이미지 feature grid (키 좌표용; positional encoding 용도 X)
        image_plane = generate_grid(feat_height, feat_width)         # [1,3,h,w]
        image_plane[:, 0] *= image_width
        image_plane[:, 1] *= image_height
        self.register_buffer('image_plane', image_plane, persistent=False)

        # 값/키용 feature projection
        self.val_proj = nn.Sequential(
            nn.BatchNorm2d(feat_dim),
            nn.ReLU(),
            nn.Conv2d(feat_dim, dim, 1, bias=False),
        )

        if no_image_features:
            self.key_proj = None
        else:
            self.key_proj = nn.Sequential(
                nn.BatchNorm2d(feat_dim),
                nn.ReLU(),
                nn.Conv2d(feat_dim, dim, 1, bias=False),
            )

        self.cross_attend = CrossAttentionEAF(dim, heads, dim_head, qkv_bias)
        self.skip = skip

        # EAF params
        self.eaf_lambda = eaf_lambda
        self.eaf_zmax = eaf_zmax
        self.use_adaptive_lambda = use_adaptive_lambda
        self.min_sigma = min_sigma
        self.max_sigma = max_sigma

        # Store original image dimensions for proper visibility scaling
        self.image_height = image_height
        self.image_width = image_width
        self.feat_height = feat_height
        self.feat_width = feat_width

    # -------- Adaptive λ_{q,i} --------
    @torch.no_grad()  # No gradients needed for geometric computation
    def _compute_lambda_qi(self, bev: BEVEmbedding, E: torch.Tensor):
        """
        Adaptive λ_{q,i}
        bev.grid: [3, H_bev, W_bev] (ego-plane)
        E:        [B, N, 4, 4]      (world->cam extrinsics)

        return: [B, N, Q, 1]
        """
        device = E.device
        grid = bev.grid.to(device)
        _, H_bev, W_bev = grid.shape
        Q = H_bev * W_bev

        xy = rearrange(grid[:2], 'c H W -> (H W) c')  # [Q,2]

        # Compute E_inv (cam->world) to get camera position
        E_inv = torch.linalg.inv(E)
        cam_center = E_inv[..., :3, 3]                # [B,N,3], cam in world
        cam_xy = cam_center[..., :2]                  # [B,N,2]

        # [B,N,Q,2]
        vec = xy.unsqueeze(0).unsqueeze(0) - cam_xy.unsqueeze(2)
        dist = torch.norm(vec, dim=-1) + 1e-6         # [B,N,Q]

        # 카메라별 최대 거리로 정규화 (권장)
        dist_max = dist.amax(dim=-1, keepdim=True)    # [B, N, 1]
        dist_norm = (dist / (dist_max + 1e-6)).clamp(0., 1.)

        # 가까울수록 σ 큼(완만), 멀수록 σ 작음(날카로움)
        sigma_qi = self.max_sigma - dist_norm * (self.max_sigma - self.min_sigma)
        lambda_qi = 1.0 / (sigma_qi + 1e-6)

        return lambda_qi.unsqueeze(-1)                # [B,N,Q,1]

    # -------- EAF W_{q,k} 계산 --------
    @torch.no_grad()  # EAF weights are pure geometric - no gradients needed
    def _compute_eaf_weights(self, bev: BEVEmbedding, I, E):
        """
        I: [B, N, 3, 3] intrinsics
        E: [B, N, 4, 4] extrinsics (world->cam)

        Returns:
          W_logits: [B, Q, NK]  with NK = N * (feat_h * feat_w)
          vis_flat: [B, Q, NK]  visibility mask
        """
        device = I.device
        B, N, _, _ = I.shape

        grid = bev.grid.to(device)
        _, H_bev, W_bev = grid.shape
        Q = H_bev * W_bev

        xy = rearrange(grid[:2], 'c H W -> (H W) c')          # [Q,2]
        ones = torch.ones(Q, 1, device=device, dtype=grid.dtype)
        zeros = torch.zeros(Q, 1, device=device, dtype=grid.dtype)

        # 두 점으로 epipolar line 정의 (z=0, z=zmax)
        P0 = torch.cat([xy, zeros, ones], dim=-1)            # [Q,4]
        P1 = torch.cat([xy, self.eaf_zmax * ones, ones], dim=-1)

        P0 = P0.unsqueeze(0).unsqueeze(0).expand(B, N, Q, 4) # [B,N,Q,4]
        P1 = P1.unsqueeze(0).unsqueeze(0).expand(B, N, Q, 4)

        # world->cam: apply E directly (no inverse needed, E is already world->cam)
        P0_cam = torch.einsum('bnij,bnqj->bnqi', E, P0.contiguous())
        P1_cam = torch.einsum('bnij,bnqj->bnqi', E, P1.contiguous())

        # cam->pixel: compute I_inv for pixel projection
        I_inv = torch.linalg.inv(I)

        # cam->pixel: use K directly (no inverse)
        p0 = torch.einsum('bnij,bnqj->bnqi', I, P0_cam[..., :3])
        p1 = torch.einsum('bnij,bnqj->bnqi', I, P1_cam[..., :3])

        p0 = p0 / torch.clamp(p0[..., 2:3], min=1e-6)
        p1 = p1 / torch.clamp(p1[..., 2:3], min=1e-6)
        
        # epipolar line (homogeneous line)
        l = torch.cross(p0, p1, dim=-1)                      # [B,N,Q,3]
        denom = torch.clamp(torch.sqrt(l[...,0]**2 + l[...,1]**2), min=1e-8)
        l_hat = l / denom.unsqueeze(-1)                      # [B,N,Q,3]

        # 이미지 feature 좌표
        _, _, h, w = self.image_plane.shape
        K_per = h * w
        x_img = rearrange(self.image_plane, '1 c h w -> (h w) c').to(device)  # [K_per,3]

        # 거리 d_{q,i,k}: [B,N,Q,K_per]
        d = torch.einsum('bnqj,kj->bnqk', l_hat, x_img).abs()

        # Adaptive λ_{q,i}
        if self.use_adaptive_lambda:
            lambda_qi = self._compute_lambda_qi(bev, E)  # [B,N,Q,1]
        else:
            lambda_qi = torch.ones((B, N, Q, 1), device=device, dtype=d.dtype)

        lam = self.eaf_lambda
        scale = (lam * lambda_qi)**2                         # [B,N,Q,1]

        W = torch.exp(- scale * (d ** 2))                    # [B,N,Q,K_per]

        # ====== [NEW] 가시성 게이트 (다중 z-샘플링 + 정확한 스케일링) ======
        # 정확한 feat 스케일 (원본 이미지 해상도 기준)
        feat_h, feat_w = self.feat_height, self.feat_width
        scale_x = feat_w / float(self.image_width)
        scale_y = feat_h / float(self.image_height)

        # 다중 z 샘플링 (any-of 방식) - wider range for better visibility detection
        z_samples = torch.tensor([0.0, 0.5, 1.5, 3.0, self.eaf_zmax],
                                 device=device, dtype=grid.dtype)  # [Nz]
        Nz = len(z_samples)

        # BEV 좌표를 다중 z로 확장
        xy = rearrange(grid[:2], 'c H W -> (H W) c')              # [Q,2]
        ones_Q = torch.ones(Q, 1, device=device, dtype=grid.dtype)

        # [Q, Nz, 4] 형태로 구성
        xy_expand = xy.unsqueeze(1).expand(Q, Nz, 2)              # [Q,Nz,2]
        z_expand = z_samples.view(1, Nz, 1).expand(Q, Nz, 1)      # [Q,Nz,1]
        ones_expand = ones_Q.unsqueeze(1).expand(Q, Nz, 1)        # [Q,Nz,1]
        Pz = torch.cat([xy_expand, z_expand, ones_expand], dim=-1)  # [Q,Nz,4]

        # [B,N,Q,Nz,4]로 확장 후 변환
        Pz = Pz.unsqueeze(0).unsqueeze(0).expand(B, N, Q, Nz, 4)

        # 카메라 좌표계로 변환: [B,N,Q,Nz,4] (reuse E from above)
        Pz_cam = torch.einsum('bnij,bnqzj->bnqzi', E, Pz)

        # cheirality 체크
        z_cam = Pz_cam[..., 2]                                     # [B,N,Q,Nz]
        cheir = (z_cam > 0)                                        # [B,N,Q,Nz]

        # 픽셀 투영
        pz = torch.einsum('bnij,bnqzj->bnqzi', I, Pz_cam[..., :3])  # [B,N,Q,Nz,3]
        u = pz[..., 0] / torch.clamp(pz[..., 2], min=1e-6)
        v = pz[..., 1] / torch.clamp(pz[..., 2], min=1e-6)

        # 이미지 좌표 → feature 좌표
        u_feat = u * scale_x
        v_feat = v * scale_y

        # in-bounds 체크
        inb = (u_feat >= 0) & (u_feat < feat_w) & (v_feat >= 0) & (v_feat < feat_h)  # [B,N,Q,Nz]

        # any-of: 하나라도 보이면 visible
        vis_per_z = cheir & inb                                    # [B,N,Q,Nz]
        vis = vis_per_z.any(dim=-1).float()[..., None]            # [B,N,Q,1]

        # 보이지 않는 카메라는 전 픽셀 0
        W = W * vis                                                # [B,N,Q,K_per]

        # ====== [END NEW] ======

        # 카메라+픽셀 축 flatten → [B,Q,NK]
        W = rearrange(W, 'b n Q K -> b Q (n K)')
        vis_flat = rearrange(vis, 'b n Q 1 -> b Q n').repeat_interleave(K_per, dim=-1)  # [B,Q,NK]

        return W, vis_flat

    # -------- forward --------
    def forward(
        self,
        x: torch.FloatTensor,          # [B, D, H_bev, W_bev]
        bev: BEVEmbedding,
        feature: torch.FloatTensor,    # [B, N, C_in, h, w]
        I: torch.FloatTensor,          # [B, N, 3, 3] intrinsics
        E: torch.FloatTensor,          # [B, N, 4, 4] extrinsics
    ):
        B, N, C_in, h, w = feature.shape

        # 키/값 feature proj
        feat_flat = rearrange(feature, 'b n c h w -> (b n) c h w')
        val_flat = self.val_proj(feat_flat)                  # (B*N, D, h, w)

        if self.key_proj is not None:
            key_flat = self.key_proj(feat_flat)              # (B*N, D, h, w)
        else:
            key_flat = val_flat

        # reshape back to [B,N,D,h,w]
        v = rearrange(val_flat, '(b n) d h w -> b n d h w', b=B, n=N)
        k = rearrange(key_flat, '(b n) d h w -> b n d h w', b=B, n=N)

        # EAF weights with visibility
        W_logits, vis_flat = self._compute_eaf_weights(bev, I, E)  # [B,Q,NK], [B,Q,NK]

        # Cross-attention (no positional enc; only EAF guides)
        out = self.cross_attend(
            q=x,
            k=k,
            v=v,
            W_logits=W_logits,
            vis_flat=vis_flat,
            skip=x if self.skip else None,
        )
        return out

    # -------- W&B Logging Methods --------
    @torch.no_grad()
    def log_lambda_qi(
        self,
        bev: BEVEmbedding,
        E_inv: torch.Tensor,
        step: int,
        sample: int = 5,
        tag: str = "lambda_qi"
    ):
        """
        Log adaptive lambda statistics and samples to W&B.

        Args:
            bev: BEV embedding containing grid
            E_inv: [B, N, 4, 4] extrinsics inverse (cam->world)
            step: current training step
            sample: number of random samples to log as text
            tag: W&B logging tag prefix
        """
        if not WANDB_AVAILABLE:
            return

        device = E_inv.device
        grid = bev.grid.to(device)
        _, Hb, Wb = grid.shape
        Q = Hb * Wb

        xy = rearrange(grid[:2], 'c H W -> (H W) c')   # [Q,2]
        cam_center = E_inv[..., :3, 3]                 # [B,N,3]
        cam_xy = cam_center[..., :2]                   # [B,N,2]

        B, N = cam_xy.shape[:2]
        b = 0  # 첫 배치만 로깅

        vec = xy.unsqueeze(0).unsqueeze(0) - cam_xy.unsqueeze(2)   # [B,N,Q,2]
        dist = torch.norm(vec, dim=-1) + 1e-6                      # [B,N,Q]

        # 카메라별 최대 거리로 정규화 (권장)
        dist_max = dist.amax(dim=-1, keepdim=True)                 # [B, N, 1]
        dist_norm = (dist / (dist_max + 1e-6)).clamp(0., 1.)
        sigma_qi = self.max_sigma - dist_norm * (self.max_sigma - self.min_sigma)
        lambda_qi = 1.0 / (sigma_qi + 1e-6)

        # W&B: 히스토그램/통계
        d0 = dist[b].flatten().detach().cpu().numpy()
        s0 = sigma_qi[b].flatten().detach().cpu().numpy()
        l0 = lambda_qi[b].flatten().detach().cpu().numpy()

        # Spearman correlation
        from scipy.stats import spearmanr
        rho_dist_sigma, _ = spearmanr(d0, s0)  # 기대: 음수 (먼 거리 -> 작은 sigma)
        rho_dist_lambda, _ = spearmanr(d0, l0)  # 기대: 양수 (먼 거리 -> 큰 lambda)

        wandb.log({
            f"{tag}/dist_hist": wandb.Histogram(d0),
            f"{tag}/sigma_hist": wandb.Histogram(s0),
            f"{tag}/lambda_hist": wandb.Histogram(l0),
            f"{tag}/dist_min": float(d0.min()),
            f"{tag}/dist_median": float(d0[len(d0)//2]),  # approximate median
            f"{tag}/dist_max": float(d0.max()),
            f"{tag}/sigma_min": float(s0.min()),
            f"{tag}/sigma_median": float(s0[len(s0)//2]),
            f"{tag}/sigma_max": float(s0.max()),
            f"{tag}/lambda_min": float(l0.min()),
            f"{tag}/lambda_median": float(l0[len(l0)//2]),
            f"{tag}/lambda_max": float(l0.max()),
            f"{tag}/spearman_dist_sigma": float(rho_dist_sigma),
            f"{tag}/spearman_dist_lambda": float(rho_dist_lambda),
            "step": step,
        })

        # 샘플 페어 몇 개 텍스트 로그
        i_idx = torch.randint(0, N, (sample,), device=device)
        q_idx = torch.randint(0, Q, (sample,), device=device)
        lines = []
        for s in range(sample):
            i = int(i_idx[s])
            q = int(q_idx[s])
            d = float(dist[b, i, q])
            sg = float(sigma_qi[b, i, q])
            lam = float(lambda_qi[b, i, q])
            lines.append(f"cam={i:02d}, q={q:05d} | dist={d:.3f} -> sigma={sg:.3f} -> lambda={lam:.3f}")

        wandb.log({f"{tag}/samples": "\n".join(lines), "step": step})

    @torch.no_grad()
    def log_epipolar_diagnostics(
        self,
        bev: BEVEmbedding,
        I_inv: torch.Tensor,
        E_inv: torch.Tensor,
        step: int,
        num_samples: int = 100,
        tag: str = "epipolar_diag"
    ):
        """
        Log epipolar line diagnostics: frustum checks, depth checks, etc.

        Args:
            bev: BEV embedding
            I_inv: [B, N, 3, 3] intrinsics inverse
            E_inv: [B, N, 4, 4] extrinsics inverse
            step: training step
            num_samples: number of random query samples
            tag: W&B tag prefix
        """
        if not WANDB_AVAILABLE:
            return

        device = E_inv.device
        B, N = E_inv.shape[:2]
        grid = bev.grid.to(device)
        _, H_bev, W_bev = grid.shape
        Q = H_bev * W_bev

        # Get feature resolution from image_plane
        _, _, feat_h, feat_w = self.image_plane.shape

        # Scale intrinsics from image resolution to feature resolution
        # image_plane is scaled by (image_w, image_h) at construction (lines 320-322)
        # We need to invert this to get feature coordinates
        img_plane_sample = self.image_plane[:, :, 0, 0]  # [1, 3]
        img_w = img_plane_sample[0, 0].item() * feat_w  # recover image width
        img_h = img_plane_sample[0, 1].item() * feat_h  # recover image height

        # Actually, let's use the max values from image_plane directly
        img_w = self.image_plane[0, 0].max().item()
        img_h = self.image_plane[0, 1].max().item()

        # Scale factor: feature_size / image_size
        scale_x = feat_w / max(img_w, 1e-6)
        scale_y = feat_h / max(img_h, 1e-6)

        xy = rearrange(grid[:2], 'c H W -> (H W) c')  # [Q,2]
        ones = torch.ones(Q, 1, device=device, dtype=grid.dtype)
        zeros = torch.zeros(Q, 1, device=device, dtype=grid.dtype)

        # Sample random queries
        q_samples = torch.randperm(Q, device=device)[:num_samples]
        xy_sample = xy[q_samples]  # [num_samples, 2]

        # Define epipolar line endpoints
        P0 = torch.cat([xy_sample, zeros[:num_samples], ones[:num_samples]], dim=-1)  # [num_samples, 4]
        P1 = torch.cat([xy_sample, self.eaf_zmax * ones[:num_samples], ones[:num_samples]], dim=-1)

        P0 = P0.unsqueeze(0).unsqueeze(0).expand(B, N, num_samples, 4)
        P1 = P1.unsqueeze(0).unsqueeze(0).expand(B, N, num_samples, 4)

        # Transform to camera space: ego->cam
        # E_inv is cam->ego, so we need ego->cam
        E = torch.linalg.inv(E_inv)

        P0_cam = torch.einsum('bnij,bnqj->bnqi', E, P0.contiguous())
        P1_cam = torch.einsum('bnij,bnqj->bnqi', E, P1.contiguous())

        # Check depths (Z > 0)
        z0 = P0_cam[..., 2]  # [B, N, num_samples]
        z1 = P1_cam[..., 2]

        # Project to image coordinates (image resolution)
        I = torch.linalg.inv(I_inv)

        p0 = torch.einsum('bnij,bnqj->bnqi', I, P0_cam[..., :3])
        p1 = torch.einsum('bnij,bnqj->bnqi', I, P1_cam[..., :3])

        # Normalize by depth to get (u, v) in image coordinates
        u0_img = p0[..., 0] / (torch.clamp(p0[..., 2].abs(), min=1e-6) * torch.sign(p0[..., 2] + 1e-10))
        v0_img = p0[..., 1] / (torch.clamp(p0[..., 2].abs(), min=1e-6) * torch.sign(p0[..., 2] + 1e-10))
        u1_img = p1[..., 0] / (torch.clamp(p1[..., 2].abs(), min=1e-6) * torch.sign(p1[..., 2] + 1e-10))
        v1_img = p1[..., 1] / (torch.clamp(p1[..., 2].abs(), min=1e-6) * torch.sign(p1[..., 2] + 1e-10))

        # Scale to feature coordinates
        u0 = u0_img * scale_x
        v0 = v0_img * scale_y
        u1 = u1_img * scale_x
        v1 = v1_img * scale_y

        # Check if points are in feature bounds [0, feat_w] x [0, feat_h]
        p0_in_bounds = ((u0 >= 0) & (u0 < feat_w) & (v0 >= 0) & (v0 < feat_h))
        p1_in_bounds = ((u1 >= 0) & (u1 < feat_w) & (v1 >= 0) & (v1 < feat_h))

        # Statistics (first batch only)
        b = 0
        logs = {}

        for cam_idx in range(N):
            z0_cam = z0[b, cam_idx]  # [num_samples]
            z1_cam = z1[b, cam_idx]

            pct_z0_positive = float((z0_cam > 0).float().mean() * 100)
            pct_z1_positive = float((z1_cam > 0).float().mean() * 100)

            p0_in = p0_in_bounds[b, cam_idx].float().mean() * 100
            p1_in = p1_in_bounds[b, cam_idx].float().mean() * 100
            either_in = ((p0_in_bounds[b, cam_idx] | p1_in_bounds[b, cam_idx]).float().mean() * 100)

            logs[f"{tag}/cam{cam_idx}_z0_positive_pct"] = pct_z0_positive
            logs[f"{tag}/cam{cam_idx}_z1_positive_pct"] = pct_z1_positive
            logs[f"{tag}/cam{cam_idx}_p0_inbounds_pct"] = float(p0_in)
            logs[f"{tag}/cam{cam_idx}_p1_inbounds_pct"] = float(p1_in)
            logs[f"{tag}/cam{cam_idx}_either_inbounds_pct"] = float(either_in)

        logs["step"] = step
        wandb.log(logs)


# -------- Fig.3 style visualization --------
@torch.no_grad()
def log_eaf_fig3_style(
    W_logits: torch.Tensor,     # [B, Q, NK]
    bev_grid: torch.Tensor,     # [3, H, W] BEV grid
    I: torch.Tensor,            # [B, N, 3, 3] intrinsics
    E: torch.Tensor,            # [B, N, 4, 4] extrinsics
    n_cams: int,
    feat_h: int,
    feat_w: int,
    image_width: int,
    image_height: int,
    cam_idx: int = 0,           # front camera
    q_samples=None,             # list of (x, y) coordinates
    step: int = 0,
    tag: str = "EAF_Fig3",
):
    """
    Create Fig.3-style visualization showing:
    - BEV grid with visibility mask for selected camera
    - EAF heatmaps for selected query points
    """
    if not WANDB_AVAILABLE:
        return

    import matplotlib.pyplot as plt
    import numpy as np

    if q_samples is None:
        q_samples = [(10.0, 0.0), (25.0, -3.0), (25.0, 3.0)]

    b_idx = 0
    B, Q, NK = W_logits.shape
    K_per = feat_h * feat_w

    # Reshape W_logits
    W_cam = W_logits[b_idx].view(Q, n_cams, feat_h, feat_w)[:, cam_idx]  # [Q, h, w]

    H_bev, W_bev = bev_grid.shape[-2:]
    assert Q == H_bev * W_bev

    # Get intrinsics and extrinsics for this camera
    I_cam = I[b_idx, cam_idx]  # [3, 3]
    E_cam = E[b_idx, cam_idx]  # [4, 4]


    # Transform BEV points to camera
    xy = rearrange(bev_grid[:2], 'c h w -> (h w) c')  # [Q, 2]
    zeros = torch.zeros(Q, 1, device=bev_grid.device, dtype=bev_grid.dtype)
    ones = torch.ones(Q, 1, device=bev_grid.device, dtype=bev_grid.dtype)
    Xw = torch.cat([xy, zeros, ones], dim=-1)  # [Q, 4] homogeneous

    Xc = (E_cam @ Xw.T).T[:, :3]  # [Q, 3] in camera space
    zc = Xc[:, 2]

    # Project to image
    p = (I_cam @ Xc.T).T
    u = p[:, 0] / (p[:, 2] + 1e-6)
    v = p[:, 1] / (p[:, 2] + 1e-6)

    # Check visibility (use image dimensions, not 0-1)
    ok_z = zc > 0
    ok_u = (u >= 0) & (u < image_width)
    ok_v = (v >= 0) & (v < image_height)
    vis_mask = (ok_z & ok_u & ok_v).view(H_bev, W_bev).cpu().numpy()

    # Create figure
    fig = plt.figure(figsize=(12, 6))
    gs = fig.add_gridspec(2, 3, height_ratios=[1, 1])
    ax_grid = fig.add_subplot(gs[:, 0])
    ax_im1 = fig.add_subplot(gs[0, 1])
    ax_im2 = fig.add_subplot(gs[0, 2])
    ax_im3 = fig.add_subplot(gs[1, 1])

    # Plot BEV visibility
    grid_vis = np.zeros((H_bev, W_bev), dtype=np.float32)
    grid_vis[vis_mask] = 1.0
    ax_grid.imshow(grid_vis, origin='lower', cmap='viridis', vmin=0, vmax=1)
    ax_grid.set_title(f"BEV cells visible to cam{cam_idx}")
    ax_grid.set_xlabel("x (forward)")
    ax_grid.set_ylabel("y (lateral)")

    # Find query indices for target coordinates
    xy_np = xy.cpu().numpy()

    def pick_q(x_star, y_star):
        d2 = (xy_np[:, 0] - x_star)**2 + (xy_np[:, 1] - y_star)**2
        return int(np.argmin(d2))

    targets = list(q_samples)[:3]
    axs = [ax_im1, ax_im2, ax_im3]

    for (x_star, y_star), ax in zip(targets, axs):
        q = pick_q(x_star, y_star)

        # Get the actual (x, y) ego coordinates for this query
        x_q = xy_np[q, 0]
        y_q = xy_np[q, 1]

        # Find the (row, col) in BEV grid
        # bev_grid has shape [3, H, W], flattened to [Q, 3] with row-major order
        r = q // W_bev
        c = q % W_bev

        # Debug: print to verify coordinates match
        # print(f"Target: ({x_star:.1f}, {y_star:.1f}) -> q={q} at grid (r={r}, c={c}) with ego ({x_q:.1f}, {y_q:.1f})")

        # Highlight on BEV grid
        # Note: imshow with origin='lower' means row=0 is at bottom
        # matplotlib Rectangle uses (x, y) = (col, row) coordinates
        rect = plt.Rectangle((c - 0.5, r - 0.5), 1, 1,
                            fill=False, edgecolor='red', lw=2)
        ax_grid.add_patch(rect)

        # Show EAF heatmap
        heat = W_cam[q].cpu().numpy()  # [h, w]
        im = ax.imshow(heat, origin='upper', cmap='magma', vmin=0, vmax=1)
        ax.set_title(f"({x_star:.1f}m, {y_star:.1f}m)")
        ax.set_xticks([])
        ax.set_yticks([])
        plt.colorbar(im, ax=ax, fraction=0.046)

    plt.tight_layout()

    # Log to W&B
    wandb.log({f"{tag}/cam{cam_idx}": wandb.Image(fig), "step": step})
    plt.close(fig)


# -------- Utility function for EAF visualization --------
@torch.no_grad()
def log_eaf_maps_to_wandb(
    W_logits: torch.Tensor,   # [B, Q, NK]
    n_cams: int,
    feat_h: int,
    feat_w: int,
    q_indices,                # int or List[int]
    step: int,
    tag: str = "EAF",
    vmax_auto: bool = True,
    q_coords=None,            # Optional list of (x, y) coordinates
):
    """
    Log EAF attention weight maps to W&B as images.

    Args:
        W_logits: [B, Q, NK] EAF weights
        n_cams: number of cameras
        feat_h: feature map height
        feat_w: feature map width
        q_indices: single int or list of BEV query indices to visualize
        step: training step
        tag: W&B logging tag prefix
        vmax_auto: if True, normalize each map to [0,1] by its max value
        q_coords: optional list of (x, y) coordinates for each query
    """
    if not WANDB_AVAILABLE:
        return

    if isinstance(q_indices, int):
        q_indices = [q_indices]

    B, Q, NK = W_logits.shape
    assert NK == n_cams * (feat_h * feat_w), f"NK mismatch: {NK} != {n_cams}*{feat_h}*{feat_w}"
    K_per = feat_h * feat_w

    # 첫 배치만
    Wb = W_logits[0].detach().cpu()                     # [Q, NK]
    Wb = Wb.view(Q, n_cams, K_per).view(Q, n_cams, feat_h, feat_w)  # [Q,N,h,w]

    logs = {}
    for idx, q in enumerate(q_indices):
        if q >= Q:
            continue  # skip invalid indices
        maps = Wb[q]                                    # [N,h,w]
        vmax = float(maps.max().clamp_min(1e-8)) if vmax_auto else None

        # Build tag with coordinates if available
        if q_coords is not None and idx < len(q_coords):
            x, y = q_coords[idx]
            tag_suffix = f"x{x:.1f}_y{y:.1f}"
        else:
            tag_suffix = f"q{q:05d}"

        imgs = []
        for i in range(n_cams):
            m = maps[i]
            if vmax is not None:
                m = (m / vmax).clamp(0, 1)              # [0,1] normalize
            img = m.numpy()                              # HxW

            # Caption with coordinates if available
            if q_coords is not None and idx < len(q_coords):
                x, y = q_coords[idx]
                caption = f"({x:.1f}m, {y:.1f}m) cam={i}"
            else:
                caption = f"q={q}, cam={i}"

            imgs.append(wandb.Image(img, caption=caption))

        logs[f"{tag}/{tag_suffix}"] = imgs

    logs["step"] = step
    wandb.log(logs)


class Encoder(nn.Module):
    def __init__(
            self,
            backbone,
            cross_view: dict,
            bev_embedding: dict,
            dim: int = 128,
            middle: List[int] = [2, 2],
            scale: float = 1.0,
    ):
        super().__init__()

        self.norm = Normalize()
        self.backbone = backbone

        if scale < 1.0:
            self.down = lambda x: F.interpolate(x, scale_factor=scale, recompute_scale_factor=False)
        else:
            self.down = lambda x: x

        assert len(self.backbone.output_shapes) == len(middle)

        cross_views = list()
        layers = list()

        for feat_shape, num_layers in zip(self.backbone.output_shapes, middle):
            _, feat_dim, feat_height, feat_width = self.down(torch.zeros(feat_shape)).shape

            cva = CrossViewAttention(feat_height, feat_width, feat_dim, dim, **cross_view)
            cross_views.append(cva)

            layer = nn.Sequential(*[ResNetBottleNeck(dim) for _ in range(num_layers)])
            layers.append(layer)

        self.bev_embedding = BEVEmbedding(dim, **bev_embedding)
        self.cross_views = nn.ModuleList(cross_views)
        self.layers = nn.ModuleList(layers)

    def forward(self, batch):
        b, n, _, _, _ = batch['image'].shape

        image = batch['image'].flatten(0, 1)            # b n c h w
        I = batch['intrinsics']                         # b n 3 3 (no inverse here)
        E = batch['extrinsics']                         # b n 4 4 (no inverse here)

        features = [self.down(y) for y in self.backbone(self.norm(image))]

        x = self.bev_embedding.get_prior()              # d H W
        x = repeat(x, '... -> b ...', b=b)              # b d H W

        for cross_view, feature, layer in zip(self.cross_views, features, self.layers):
            feature = rearrange(feature, '(b n) ... -> b n ...', b=b, n=n)

            x = cross_view(x, self.bev_embedding, feature, I, E)
            x = layer(x)

        return x