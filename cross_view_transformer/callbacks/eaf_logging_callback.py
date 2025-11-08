import pytorch_lightning as pl
import torch

from typing import Any
from pytorch_lightning.utilities.types import STEP_OUTPUT
from pytorch_lightning.utilities import rank_zero_only

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
                E_inv = batch['extrinsics'].inverse()
                cross_view.log_lambda_qi(
                    bev=encoder.bev_embedding,
                    E_inv=E_inv,
                    step=global_step,
                    sample=self.num_samples,
                    tag="adaptive_lambda",
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
                    W_logits = cross_view._compute_eaf_weights(
                        bev=encoder.bev_embedding,
                        I_inv=I_inv,
                        E_inv=E_inv,
                    )  # [B, Q, NK]

                n_cams = batch['image'].shape[1]
                feat_h = cross_view.image_plane.shape[-2]
                feat_w = cross_view.image_plane.shape[-1]

                # Get BEV dimensions
                bev_h = encoder.bev_embedding.learned_features.shape[1]
                bev_w = encoder.bev_embedding.learned_features.shape[2]
                Q = bev_h * bev_w

                # Log W_logits statistics
                self._log_w_logits_stats(W_logits, global_step)

                # Select query indices to visualize (evenly spaced)
                q_indices = [0, Q//4, Q//2, 3*Q//4][:self.num_query_vis]

                log_eaf_maps_to_wandb(
                    W_logits=W_logits,
                    n_cams=n_cams,
                    feat_h=feat_h,
                    feat_w=feat_w,
                    q_indices=q_indices,
                    step=global_step,
                    tag="EAF",
                )
            except Exception as e:
                print(f"Error logging EAF maps at step {global_step}: {e}")

    @rank_zero_only
    def _log_w_logits_stats(self, W_logits: torch.Tensor, step: int):
        """
        Log W_logits (EAF weights) statistics to detect extreme values.

        Args:
            W_logits: [B, Q, NK] EAF attention weights
            step: current training step
        """
        if not WANDB_AVAILABLE:
            return

        with torch.no_grad():
            W = W_logits[0].detach().cpu()  # first batch only
            W_flat = W.flatten().numpy()

            # Basic statistics
            w_min = float(W.min())
            w_max = float(W.max())
            w_mean = float(W.mean())
            w_std = float(W.std())
            w_median = float(W.median())

            # Check for extreme values
            num_zeros = int((W < 1e-10).sum())
            num_ones = int((W > 0.99).sum())
            num_total = W.numel()

            # Entropy per query (measures how uniform the attention is)
            # H = -sum(p * log(p)), higher entropy = more uniform
            W_normalized = W / (W.sum(dim=-1, keepdim=True) + 1e-10)  # [Q, NK]
            entropy = -(W_normalized * torch.log(W_normalized + 1e-10)).sum(dim=-1)  # [Q]
            avg_entropy = float(entropy.mean())
            min_entropy = float(entropy.min())
            max_entropy = float(entropy.max())

            # Effective sparsity: what percentage of weight is concentrated in top-k%?
            sorted_W, _ = torch.sort(W, dim=-1, descending=True)
            top10_sum = sorted_W[:, :int(sorted_W.shape[1] * 0.1)].sum(dim=-1)
            avg_top10_ratio = float((top10_sum / (sorted_W.sum(dim=-1) + 1e-10)).mean())

            wandb.log({
                "W_logits/min": w_min,
                "W_logits/max": w_max,
                "W_logits/mean": w_mean,
                "W_logits/std": w_std,
                "W_logits/median": w_median,
                "W_logits/histogram": wandb.Histogram(W_flat),
                "W_logits/num_near_zero": num_zeros,
                "W_logits/num_near_one": num_ones,
                "W_logits/pct_near_zero": 100.0 * num_zeros / num_total,
                "W_logits/pct_near_one": 100.0 * num_ones / num_total,
                "W_logits/entropy_mean": avg_entropy,
                "W_logits/entropy_min": min_entropy,
                "W_logits/entropy_max": max_entropy,
                "W_logits/top10_concentration": avg_top10_ratio,
                "step": step,
            })
