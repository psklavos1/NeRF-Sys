import warnings

from data.ram_rays_dataset import RamRaysDataset

warnings.filterwarnings("ignore", category=UserWarning, module="torchvision")
import time
from pathlib import Path
from torch.amp import autocast

import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
import lpips  # Learned Perceptual Image Patch Similarity
from pytorch_msssim import ssim  # Structural similarity metrics
from imageio import imwrite

from utils import collate_rays, psnr
from utils import MetricLogger
from common.utils import to_device_tree

from nerfs.losses import compute_mse_loss
from nerfs.color_space import linear_to_srgb, color_space_transformer
from nerfs.ray_rendering import render_image
from nerfs.meta_learning import (
    task_adapt,
    runtime_adapt_inplace as runtime_adapt,
)
from common.utils import get_optimizer

def validate_nerf_model(P, model, test_loader, steps, logger):
    model.eval()
    P.fim = False  

    device = next(model.parameters()).device
    inner_lr = P.inner_lr
    iterations = int(getattr(P, "tto", getattr(P, "inner_iter", 1)))
    mixed_precision = bool(getattr(P, "mixed_precision", False))
    metric_logger = MetricLogger(delimiter="  ")
    tasks_cap = int(getattr(P, "max_test_tasks", 5))
    tasks_seen = 0

    # ------ global accumulators  ------
    support_loss_sum_global = torch.tensor(0.0, device=device)
    support_ray_count_global = 0

    query_loss_sum_global = torch.tensor(0.0, device=device)
    query_ray_count_global = 0

    for task_data in test_loader:
        batch_start_t = time.time()
        task_data = to_device_tree(task_data, device)
        cids = sorted(task_data.keys())

        # ------ per-region accumulators ------
        region_inner_sum = {cid: torch.tensor(0.0, device=device) for cid in cids}
        region_inner_count = {cid: 0 for cid in cids}
        region_query_sum = {cid: torch.tensor(0.0, device=device) for cid in cids}
        region_query_count = {cid: 0 for cid in cids}

        # ------ iterate regions and tasks ------
        for cid in cids:
            for task in task_data[cid]:
                sup_i, qry_i = task["support"], task["query"]

                # number of rays for proper weighting (matching training)
                n_sup = int(sup_i["rays"].shape[0])
                n_q = int(qry_i["rays"].shape[0])

                if n_sup == 0 or n_q == 0:
                    logger.log(f"[EVAL][WARN] Empty task in region {cid}; skipping.")
                    continue

                # Inner adaptation on SUPPORT → fast weights + inner loss trace
                fast, inner_losses = task_adapt(
                    P,
                    model,
                    sup_i,
                    inner_lr,
                    iterations,
                    active_module=cid,
                )
                inner_last = (
                    inner_losses[-1]
                    if inner_losses
                    else torch.tensor(0.0, device=device)
                )  # mean over support rays

                # Query loss AFTER adaptation (θ*) — mean per-ray 
                with torch.no_grad(), autocast(
                    device_type="cuda",
                    enabled=mixed_precision,
                    dtype=torch.float16,
                ):
                    qa = compute_mse_loss(
                        P,
                        model=model,
                        data=qry_i,
                        params=fast,
                        active_module=cid,
                        reduction="mean",
                    )

                # sample-weighted accumulators (like train_step_ray)
                support_loss_sum_global += inner_last.detach() * n_sup
                support_ray_count_global += n_sup

                query_loss_sum_global += qa.detach() * n_q
                query_ray_count_global += n_q

                region_inner_sum[cid] += inner_last.detach() * n_sup
                region_inner_count[cid] += n_sup

                region_query_sum[cid] += qa.detach() * n_q
                region_query_count[cid] += n_q

        # per-loader-batch timing (seconds per batch)
        batch_time = time.time() - batch_start_t
        metric_logger.meters["batch_time"].update(batch_time, n=1)
        metric_logger.meters["eval_context"].update(len(cids), n=1)

        tasks_seen += 1
        if tasks_seen >= tasks_cap:
            break

    # Guard against empty evaluation
    if query_ray_count_global == 0:
        logger.log("[EVAL] No valid query rays found in test_loader; returning PSNR=0.0")
        model.train(True)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return 0.0

    # ------ Global metrics (sample-weighted means) ------
    loss_in_global = (
        support_loss_sum_global / support_ray_count_global
        if support_ray_count_global > 0
        else torch.tensor(0.0, device=device)
    )
    loss_out_global = query_loss_sum_global / query_ray_count_global
    psnr_in_global = psnr(loss_in_global)
    psnr_out_global = psnr(loss_out_global)

    # ------ Metric logger ------
    metric_logger.meters["loss_in"].update(
        float(loss_in_global), n=support_ray_count_global
    )
    metric_logger.meters["psnr_in"].update(
        float(psnr_in_global), n=support_ray_count_global
    )
    metric_logger.meters["loss_out"].update(
        float(loss_out_global), n=query_ray_count_global
    )
    metric_logger.meters["psnr_out"].update(
        float(psnr_out_global), n=query_ray_count_global
    )
    metric_logger.synchronize_between_processes()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # ------ Logging ------
    logger.log(
        " * [EVAL] [LossIn %.6f] [LossOut %.6f] [PSNRIn %.3f] [PSNROut %.3f]"
        % (
            metric_logger.loss_in.global_avg,
            metric_logger.loss_out.global_avg,
            metric_logger.psnr_in.global_avg,
            metric_logger.psnr_out.global_avg,
        )
    )
    logger.scalar_summary("eval/tto", float(iterations), steps)
    logger.scalar_summary("eval/loss_in", metric_logger.loss_in.global_avg, steps)
    logger.scalar_summary("eval/loss_out", metric_logger.loss_out.global_avg, steps)
    logger.scalar_summary("eval/psnr_in", metric_logger.psnr_in.global_avg, steps)
    logger.scalar_summary("eval/psnr_out", metric_logger.psnr_out.global_avg, steps)

    return float(metric_logger.psnr_out.global_avg)

def runtime_evaluate_model(
    P,
    model,
    test_loader,  # ImageMetaDataset DataLoader
    steps,
    logger,
    scene_box,
) -> float:
    """
    Phase A: For each image batch in test_loader -> build RamRaysDataset -> shuffled support loader -> runtime_adapt (cumulative).
    Phase B: Render & score all test images with the final 'fast'.
    """
    device = next(model.parameters()).device
    use_amp = bool(getattr(P, "use_amp", False))
    num_support = int(getattr(P, "support_rays", 4096))
    ray_samples = int(getattr(P, "ray_samples", 64))
    chunk_points = int(getattr(P, "chunk_points", 1 << 16))
    metrics_space = getattr(P, "color_space", "linear")  # 'linear'|'srgb'|'identity'

    print(f"======================================= TTO: {P.tto} =======================================")
    # -------- Phase A: Adaptation over support rays built per image batch --------
    model.train(True)
    total_support = 0
    logger.log("[Phase 1] Task Adaptation on validation images")
    total_batches = 0
    
    optimizer = get_optimizer(P,model)
    t_adapt_start = time.time()

    for batch in test_loader:
        total_batches += 1
        metas = batch["metas"]  # list of meta dict/objects

        # Build RamRaysDataset directly from this image batch
        rays_ds = RamRaysDataset(
            metas,
            center_pixels=True,
            ray_gen_kwargs={
                "scene_box": scene_box,
                "near_far_override": (P.near, P.far),
            },
        )

        total_support += int(len(rays_ds))

        data_loader = DataLoader(
            rays_ds,
            batch_size=num_support,
            shuffle=True,  # IMPORTANT: randomize support rays each epoch
            drop_last=False,
            num_workers=4,  # safer in eval; increase if you like
            persistent_workers=True,
            prefetch_factor=4,
            pin_memory=True,
            collate_fn=collate_rays,
        )

        out = runtime_adapt(
            P=P,
            model=model,
            data_loader=data_loader,
            optimizer=optimizer,
            steps=steps
        )

        logger.log(
            f"[Batch {total_batches}] Rays={len(rays_ds)} last_loss={out['loss']:.6f}"
        )
        del rays_ds

    # finish timing (Phase A)
    if device.type == "cuda":
        torch.cuda.synchronize()
    adapt_time_sec = time.time() - t_adapt_start

    logger.log(
        f"[ADAPTATION END] [{adapt_time_sec:.2f}s] Total support rays seen (with shuffling by loader): {total_support} in {total_batches} Batches."
    )
    torch.cuda.empty_cache()

    # -------- Phase B: Render & metrics --------
    logger.log("[Phase 2] Rendering Images")

    model.eval()
    scorer_lpips = lpips.LPIPS(net="alex").to(device)
    meter = MetricLogger(delimiter="  ")

    out_root_rendered = Path("logs/eval") / Path(P.fname) / "rendered"
    out_pred = out_root_rendered / ("pred" + str(getattr(P, "tto", steps)))
    out_gt = out_root_rendered / "gt"
    out_gt.mkdir(parents=True, exist_ok=True)
    out_pred.mkdir(parents=True, exist_ok=True)

    idx = 0
    t0 = time.time()
    for batch in test_loader:
        metas = batch["metas"]
        rgbs = batch["rgbs_raw"]

        for b, md in enumerate(metas):
            H, W = int(md.H), int(md.W)
            fx, fy, cx, cy = md.intrinsics.flatten().tolist()[:4]
            c2w = md.c2w
            gt_img_srgb = (rgbs[b].float() / 255.0).clamp_(0, 1).to(device)  # HxWx3

            with torch.cuda.amp.autocast(enabled=use_amp, dtype=torch.float16):
                pred_lin, _, _ = render_image(
                    model,
                    H=H,
                    W=W,
                    fx=fx,
                    fy=fy,
                    cx=cx,
                    cy=cy,
                    c2w=c2w,
                    scene_box=scene_box,
                    ray_samples=ray_samples,
                    bg_color_default=getattr(P, "bg_color_default", "white"),
                    chunk_points=chunk_points,
                    center_pixels=True,
                    use_amp=use_amp,
                )

            # ---- metrics
            pred_bchw_lin = pred_lin.permute(2, 0, 1).unsqueeze(0)
            gt_bchw_srgb = gt_img_srgb.permute(2, 0, 1).unsqueeze(0)
            pred_bchw, gt_bchw = color_space_transformer(
                pred_bchw_lin, gt_bchw_srgb, metrics_space
            )

            mse = F.mse_loss(pred_bchw, gt_bchw, reduction="mean")
            psnr_val = (-10.0 * torch.log10(mse.clamp_min(1e-8))).item()
            ssim_val = ssim(pred_bchw, gt_bchw, data_range=1.0).mean().item()

            # LPIPS on sRGB in [-1,1]
            if metrics_space == "srgb":
                pred_srgb_img = pred_bchw  # NCHW
            else:
                pred_srgb_img = (
                    linear_to_srgb(pred_lin).permute(2, 0, 1).unsqueeze(0)
                )  # NCHW

            pred_lp = pred_srgb_img * 2 - 1  # NCHW in [-1,1]
            gt_lp = gt_bchw_srgb * 2 - 1  # NCHW in [-1,1]
            lpips_val = scorer_lpips(pred_lp, gt_lp).mean().item()

            # log/save
            meter.meters["psnr"].update(psnr_val, n=1)
            meter.meters["ssim"].update(ssim_val, n=1)
            meter.meters["lpips"].update(lpips_val, n=1)
            logger.scalar_summary("eval_image/psnr", psnr_val, idx)
            logger.scalar_summary("eval_image/ssim", ssim_val, idx)
            logger.scalar_summary("eval_image/lpips", lpips_val, idx)

            logger.log(
                " * [IMG %d]  PSNR %.3f | SSIM %.3f | LPIPS %.3f"
                % (idx + 1, psnr_val, ssim_val, lpips_val)
            )

            pred_srgb = (
                ((pred_srgb_img.squeeze(0).permute(1, 2, 0).clamp(0, 1) * 255.0) + 0.5)
                .byte()
                .cpu()
                .numpy()
            )
            gt_srgb = (gt_img_srgb.clamp(0, 1) * 255.0 + 0.5).byte().cpu().numpy()
            imwrite(out_pred / f"{idx:06d}.png", pred_srgb)
            imwrite(out_gt / f"{idx:06d}.png", gt_srgb)

            meter.meters["batch_time"].update(time.time() - t0, n=1)
            t0 = time.time()
            idx += 1

    meter.synchronize_between_processes()
    logger.log(
        " * [RENDERING END] imgs %d | PSNR %.3f | SSIM %.3f | LPIPS %.3f"
        % (idx, meter.psnr.global_avg, meter.ssim.global_avg, meter.lpips.global_avg)
    )
    logger.scalar_summary("eval/psnr", meter.psnr.global_avg, num_support)
    logger.scalar_summary("eval/ssim", meter.ssim.global_avg, num_support)
    logger.scalar_summary("eval/lpips", meter.lpips.global_avg, num_support)
    metrics = {"psnr": meter.psnr.global_avg, "ssim": meter.ssim.global_avg, "lpips": meter.lpips.global_avg, "duration": float(adapt_time_sec)}
    return metrics
