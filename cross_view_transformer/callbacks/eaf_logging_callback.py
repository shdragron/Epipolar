import pytorch_lightning as pl
import torch

from typing import Any
from pytorch_lightning.utilities.types import STEP_OUTPUT
from pytorch_lightning.utilities import rank_zero_only
from einops import rearrange

from cross_view_transformer.model.encoder import log_eaf_maps_to_wandb

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


class EAFLoggingCallback(pl.Callback):
    """
    Callback to log EAF (Epipolar Attention Field) statistics and visualizations to W&B.

    Logs:
    - Adaptive lambda statistics (distributions, min/max/median)
    - EAF attention weight maps as images
    """

    def __init__(
        self,
        lambda_log_interval: int = 200,
        eaf_vis_interval: int = 400,
        num_samples: int = 8,
        num_query_vis: int = 4,
    ):
        super().__init__()
        self.lambda_log_interval = lambda_log_interval
        self.eaf_vis_interval = eaf_vis_interval
        self.num_samples = num_samples
        self.num_query_vis = num_query_vis

    
    @staticmethod
    def _xy_in_bev_bounds(grid: torch.Tensor, xy: tuple) -> bool:
        """
        grid: [3,H,W] (world coords), xy: (x_target, y_target) in meters
        """
        x_t, y_t = xy
        gx, gy = grid[0], grid[1]              # [H,W]
        # grid는 torch.Tensor이므로 .item() 없이 비교 가능
        return (gx.min() <= x_t <= gx.max()) and (gy.min() <= y_t <= gy.max())
    
    @rank_zero_only
    def on_train_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs: STEP_OUTPUT,
        batch: Any,
        batch_idx: int,
        *args,
        **kwargs
    ) -> None:
        global_step = trainer.global_step

        # Get the model's encoder (backbone is the actual model)
        if not hasattr(pl_module, 'backbone') or not hasattr(pl_module.backbone, 'encoder'):
            return

        encoder = pl_module.backbone.encoder

        # Check if encoder has cross_views
        if not hasattr(encoder, 'cross_views') or len(encoder.cross_views) == 0:
            return

        # Use the first cross-view layer for logging
        cross_view = encoder.cross_views[0]

        # (1) Log adaptive lambda statistics
        if global_step % self.lambda_log_interval == 0 and global_step > 0:
            try:
                I_inv = batch['intrinsics'].inverse()
                E_inv = batch['extrinsics'].inverse()

                cross_view.log_lambda_qi(
                    bev=encoder.bev_embedding,
                    E_inv=E_inv,
                    step=global_step,
                    sample=self.num_samples,
                    tag="adaptive_lambda",
                )

                # Log epipolar diagnostics
                cross_view.log_epipolar_diagnostics(
                    bev=encoder.bev_embedding,
                    I_inv=I_inv,
                    E_inv=E_inv,
                    step=global_step,
                    num_samples=100,
                    tag="epipolar_diag",
                )
            except Exception as e:
                print(f"Error logging lambda_qi at step {global_step}: {e}")

        # (2) Log EAF attention weight maps
        if global_step % self.eaf_vis_interval == 0 and global_step > 0:
            try:
                I_inv = batch['intrinsics'].inverse()
                E_inv = batch['extrinsics'].inverse()

                # Compute EAF weights
                with torch.no_grad():
                    out = cross_view._compute_eaf_weights(
                    bev=encoder.bev_embedding,
                    I_inv=I_inv,
                    E_inv=E_inv,
                    )
                    W_logits = out[0] if isinstance(out, tuple) else out
                    
                n_cams = batch['image'].shape[1]
                feat_h = cross_view.image_plane.shape[-2]
                feat_w = cross_view.image_plane.shape[-1]

                # Select query indices by real BEV coordinates (not flat indices)
                targets_xy = [
                    # ---- Front band (정면) ----
                    (8.0,  0.0),  (15.0,  0.0),  (25.0,  0.0),  (40.0,  0.0),
                    (15.0,  3.0), (15.0, -3.0), (25.0,  6.0), (25.0, -6.0),

                    # ---- Front-Left / Front-Right (전측면) ----
                    (12.0,  8.0), (20.0, 12.0), (28.0, 16.0),
                    (12.0, -8.0), (20.0,-12.0), (28.0,-16.0),

                    # ---- Side far (측면 멀리; 차량 바로 옆/사선) ----
                    (5.0,  20.0), (10.0, 25.0), (5.0, -20.0), (10.0, -25.0),

                    # ---- Back band (후방) ----
                    (-8.0,  0.0), (-15.0,  0.0), (-25.0,  0.0), (-40.0,  0.0),
                    (-15.0,  4.0), (-15.0, -4.0),

                    # ---- Back-Left / Back-Right (후측면) ----
                    (-12.0,  8.0), (-20.0, 12.0),
                    (-12.0, -8.0), (-20.0,-12.0),
                ]
                # 그리드 범위 밖 좌표는 필터(안전)
                targets_xy = [p for p in targets_xy if self._xy_in_bev_bounds(encoder.bev_embedding.grid, p)]
                q_indices = self._pick_queries_by_xy(encoder.bev_embedding.grid, targets_xy)

                log_eaf_maps_to_wandb(
                    W_logits=W_logits,
                    n_cams=n_cams, feat_h=feat_h, feat_w=feat_w,
                    q_indices=q_indices, q_coords=targets_xy,
                    step=global_step, tag="EAF",
                )

                # Log Fig.3 style visualization (front camera only)
                from cross_view_transformer.model.encoder import log_eaf_fig3_style
                log_eaf_fig3_style(
                    W_logits=W_logits,
                    bev_grid=encoder.bev_embedding.grid,
                    I_inv=I_inv,
                    E_inv=E_inv,
                    n_cams=n_cams,
                    feat_h=feat_h,
                    feat_w=feat_w,
                    cam_idx=1,  # CAM_FRONT (nuScenes: 0=FL, 1=F, 2=FR)
                    q_samples=[(10.0, 0.0), (25.0, -3.0), (25.0, 3.0)],
                    step=global_step,
                    tag="EAF_Fig3",
                )
            except Exception as e:
                print(f"Error logging EAF maps at step {global_step}: {e}")

    def _pick_queries_by_xy(self, grid: torch.Tensor, targets_xy: list):
        """
        Pick query indices based on real BEV world coordinates.

        Args:
            grid: [3, H, W] BEV grid in world coordinates
            targets_xy: list of (x, y) coordinates in meters

        Returns:
            list of query indices
        """
        # grid[:2] is [2, H, W] with (x, y) coordinates
        xy = rearrange(grid[:2], 'c H W -> (H W) c')  # [Q, 2]

        q_list = []
        for x_target, y_target in targets_xy:
            # Find closest grid point to target
            dist_sq = (xy[:, 0] - x_target)**2 + (xy[:, 1] - y_target)**2
            q = int(dist_sq.argmin().item())
            q_list.append(q)

        return q_list
