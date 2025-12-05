import torch
import torch.nn.functional as F

import lpips  # Learned Perceptual Image Patch Similarity
from pytorch_msssim import ms_ssim, ssim  # Structural similarity metrics

from train.gradient_based import inner_adapt
from utils import MetricLogger, psnr, get_meta_batch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def check(P):
    """
    Dummy placeholder function (currently just returns True).
    Could be used to control date-based evaluation logic.
    """
    filename_with_today_date = True
    return filename_with_today_date


def test_model(P, wrapper, loader, steps, logger=None):
    """
    Evaluates a meta-learned model on unseen tasks using inner-loop adaptation,
    then computes perceptual and structural quality metrics.

    Args:
        P (argparse.Namespace): Hyperparameters and configuration settings.
        wrapper (nn.Module): Meta-learning model wrapper (MetaWrapper).
        loader (DataLoader): DataLoader yielding new meta-test tasks.
        steps (int): Number of inner-loop adaptation steps.
        logger (Logger, optional): Optional logger for recording outputs.

    Returns:
        float: Final PSNR average across all test tasks.
    """
    metric_logger = MetricLogger(delimiter="  ")

    if logger is None:
        log_ = print
    else:
        log_ = logger.log

    # Put the model into evaluation mode and initialize coordinates (for INR models)
    mode = wrapper.training
    wrapper.eval()
    wrapper.coord_init()

    lpips_score = lpips.LPIPS(net="alex").to(
        device
    )  # LPIPS setup for perceptual metric

    # Choose appropriate inner-loop adaptation method
    kwargs = {}
    if P.algo == "maml_full_evaluate_gradscale":
        adapt = inner_adapt
        kwargs["sample_type"] = P.sample_type
        kwargs["scale_type"] = "grad"
    else:
        adapt = inner_adapt

    # Loop through tasks in the test loader
    for n, task_data in enumerate(loader):
        task_data = {k: v.to(device, non_blocking=True) for k, v in task_data.items()}

        # Unpack and format meta-batch
        batch_size, context = get_meta_batch(P, task_data)

        # Adapt parameters to the current task using inner-loop steps
        params = adapt(
            wrapper,
            context,
            P.inner_lr,
            P.num_submodules,
            first_order=True,
            train_mode=False,
            **kwargs,
        )[0]

        # Forward pass with adapted parameters
        with torch.no_grad():
            pred = wrapper(None, params).clamp(0, 1)

        # Compute evaluation metrics
        # TODO: Add ray
        if P.data_type == "img":
            context = context[0]  # Use target image
            lpips_result = lpips_score((pred * 2 - 1), (context * 2 - 1)).mean()
            psnr_result = psnr(
                F.mse_loss(
                    context.view(batch_size, -1),
                    pred.view(batch_size, -1),
                    reduce=False,
                ).mean(dim=1)
            ).mean()
            ssim_result = ssim(pred, context, data_range=1.0).mean()
            log_ssim_result = (
                -10.0 * torch.log10(1 - ssim(pred, context, data_range=1.0) + 1e-24)
            ).mean()
        else:
            raise NotImplementedError(
                "Evaluation for this data type is not implemented."
            )

        # Update metric logger
        metric_logger.meters["lpips_result"].update(lpips_result.item(), n=batch_size)
        metric_logger.meters["psnr_result"].update(psnr_result.item(), n=batch_size)
        metric_logger.meters["ssim_result"].update(ssim_result.item(), n=batch_size)
        metric_logger.meters["log_ssim_result"].update(
            log_ssim_result.item(), n=batch_size
        )

        # Periodically print metrics
        if n % 10 == 0:
            metric_logger.synchronize_between_processes()
            log_(
                f" * [EVAL {n}] [PSNR %.3f] [LPIPS %.3f] "
                "[SSIM %.3f] [LOG SSIM %.3f] "
                % (
                    metric_logger.psnr_result.global_avg,
                    metric_logger.lpips_result.global_avg,
                    metric_logger.ssim_result.global_avg,
                    metric_logger.log_ssim_result.global_avg,
                )
            )

    # Final metric report after all tasks
    metric_logger.synchronize_between_processes()
    log_(
        " * [EVAL] [PSNR %.3f] [LPIPS %.3f] "
        "[SSIM %.3f] [LOG SSIM %.3f] "
        % (
            metric_logger.psnr_result.global_avg,
            metric_logger.lpips_result.global_avg,
            metric_logger.ssim_result.global_avg,
            metric_logger.log_ssim_result.global_avg,
        )
    )

    # Restore training mode and clean up
    wrapper.train(mode)
    torch.cuda.empty_cache()

    return metric_logger.psnr_result.global_avg
