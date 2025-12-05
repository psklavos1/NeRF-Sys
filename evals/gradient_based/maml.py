import torch
from train.gradient_based.__init__ import inner_adapt
from utils import MetricLogger, psnr, get_meta_batch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def check(P):
    """
    Placeholder function returning a static True value.
    Could be extended to add date-based evaluation logic.
    """
    filename_with_today_date = True
    return filename_with_today_date


def test_model_img(P, wrapper, loader, steps, logger=None):
    """
    Evaluates the model on image-based tasks (meta-test phase).
    Performs inner-loop adaptation and reports patch-wise loss and PSNR statistics.

    Args:
        P (argparse.Namespace): Configuration and hyperparameters.
        wrapper (nn.Module): Meta-learned model.
        loader (DataLoader): Task-level data loader.
        steps (int): Outer-loop step (for logging).
        logger (Logger, optional): Optional logger for logging and TensorBoard.

    Returns:
        float: Final PSNR after adaptation (averaged across tasks).
    """
    metric_logger = MetricLogger(delimiter="  ")

    log_ = logger.log if logger else print

    mode = wrapper.training  # store model mode
    wrapper.eval()
    wrapper.coord_init()  # initialize coordinates if using implicit models

    for n, task_data in enumerate(loader):
        task_data = {k: v.to(device, non_blocking=True) for k, v in task_data.items()}
        batch_size, episode_batch = get_meta_batch(P, task_data)

        # Perform inner-loop adaptation
        params, loss_in, loss_in_log, res_in, grad_in = inner_adapt(
            P,
            wrapper,
            episode_batch,
            P.trained_inner_lr,
            P.num_submodules,
            first_order=True,
            order=P.order,
        )

        # Reshape losses to summarize them per-patch
        loss_in = loss_in.view(P.num_submodules, P.tto, batch_size, -1).mean(dim=-1)[
            -1
        ][-1]
        loss_in_log = loss_in_log.view(
            P.num_submodules, P.num_submodules, batch_size, -1
        ).mean(dim=-1)

        # Evaluate model with adapted parameters
        with torch.no_grad():
            loss_out, res_out = wrapper(episode_batch, params=params)
            loss_out = loss_out.view(batch_size, -1).mean(dim=-1)

        # Logging task-level statistics
        metric_logger.meters["loss_in"].update(loss_in.mean().item(), n=batch_size)
        metric_logger.meters["psnr_in"].update(
            psnr(loss_in).mean().item(), n=batch_size
        )
        metric_logger.meters["loss_out"].update(loss_out.mean().item(), n=batch_size)
        metric_logger.meters["psnr_out"].update(
            psnr(loss_out).mean().item(), n=batch_size
        )
        metric_logger.meters["eval_context"].update(episode_batch, n=batch_size)

        if n * P.test_batch_size > P.max_test_task:
            break

    metric_logger.synchronize_between_processes()
    log_(
        " * [EVAL] [LossIn %.3f] [LossOut %.3f] [PSNRIn %.3f] [PSNROut %.3f]"
        % (
            metric_logger.loss_in.global_avg,
            metric_logger.loss_out.global_avg,
            metric_logger.psnr_in.global_avg,
            metric_logger.psnr_out.global_avg,
        )
    )

    # Log scalar and image summaries if logger is used
    if logger:
        if P.log_method == "step":
            for i in range(P.num_submodules):
                for j in range(P.num_submodules):
                    logger.writer.add_scalar(
                        f"eval_loss_in_step{i:02}/loss_patch{j:02}",
                        loss_in_log[i][j].mean().item(),
                        steps,
                    )
                    logger.writer.add_scalar(
                        f"eval_psnr_in_step{i:02}/psnr_patch{j:02}",
                        psnr(loss_in_log[i][j]).mean().item(),
                        steps,
                    )
                if i > 0:
                    logger.writer.add_scalar(
                        f"eval_loss_in_step{i:02}",
                        loss_in_log[i][:i].mean().item(),
                        steps,
                    )
                    logger.writer.add_scalar(
                        f"eval_psnr_in_step{i:02}",
                        psnr(loss_in_log[i][:i]).mean().item(),
                        steps,
                    )
        elif P.log_method == "patch":
            for i in range(P.num_submodules):
                for j in range(P.num_submodules):
                    logger.writer.add_scalar(
                        f"eval_loss_in_patch{i:02}/loss_step{j:02}",
                        loss_in_log[j][i].mean().item(),
                        steps,
                    )
                    logger.writer.add_scalar(
                        f"eval_psnr_in_patch{i:02}/psnr_step{j:02}",
                        psnr(loss_in_log[j][i]).mean().item(),
                        steps,
                    )

        logger.scalar_summary("eval/inner_lr", P.trained_inner_lr, steps)
        logger.scalar_summary("eval/loss_in", metric_logger.loss_in.global_avg, steps)
        logger.scalar_summary("eval/loss_out", metric_logger.loss_out.global_avg, steps)
        logger.scalar_summary("eval/psnr_in", metric_logger.psnr_in.global_avg, steps)
        logger.scalar_summary("eval/psnr_out", metric_logger.psnr_out.global_avg, steps)
        # if P.data_type == "img":
        #     logger.image_summary(
        #         "eval/imgs_in", res_in, metric_logger.eval_context, steps
        #     )
        #     logger.image_summary(
        #         "eval/imgs_out", res_out, metric_logger.eval_context, steps
        #     )

    wrapper.train(mode)
    return metric_logger.psnr_out.global_avg


def validate_nerf_model(P, wrapper, loader, steps, logger=None):
    pass


def test_model_video(P, wrapper, loader, steps, logger=None):
    """
    Same as test_model_img but customized for video-based tasks.

    Args:
        P (argparse.Namespace): Hyperparameters and configuration.
        wrapper (nn.Module): Meta-learning model.
        loader (DataLoader): DataLoader providing meta-test tasks (video data).
        steps (int): Training step for logging.

    Returns:
        float: Average PSNR across evaluated video tasks.
    """
    # The body is identical to test_model_img except for `logger.video_summary()` at the end.
    metric_logger = MetricLogger(delimiter="  ")
    log_ = logger.log if logger else print

    mode = wrapper.training
    wrapper.eval()
    wrapper.coord_init()

    for n, task_data in enumerate(loader):
        task_data = {k: v.to(device, non_blocking=True) for k, v in task_data.items()}
        batch_size, context = get_meta_batch(P, task_data)
        params, loss_in, loss_in_log, res_in, grad_in = inner_adapt(
            P,
            wrapper,
            context[0],
            P.trained_inner_lr,
            P.num_submodules,
            first_order=True,
            order=P.order,
        )
        loss_in = loss_in.view(P.num_submodules, P.tto, batch_size, -1).mean(dim=-1)[
            -1
        ][-1]
        loss_in_log = loss_in_log.view(
            P.num_submodules, P.num_submodules, batch_size, -1
        ).mean(dim=-1)

        with torch.no_grad():
            loss_out, res_out = wrapper(context[0], params=params)
            loss_out = loss_out.view(batch_size, -1).mean(dim=-1)

        metric_logger.meters["loss_in"].update(loss_in.mean().item(), n=batch_size)
        metric_logger.meters["psnr_in"].update(
            psnr(loss_in).mean().item(), n=batch_size
        )
        metric_logger.meters["loss_out"].update(loss_out.mean().item(), n=batch_size)
        metric_logger.meters["psnr_out"].update(
            psnr(loss_out).mean().item(), n=batch_size
        )
        metric_logger.meters["eval_context"].update(context[0], n=batch_size)

        if n * P.test_batch_size > P.max_test_task:
            break

    metric_logger.synchronize_between_processes()
    log_(
        " * [EVAL] [LossIn %.3f] [LossOut %.3f] [PSNRIn %.3f] [PSNROut %.3f]"
        % (
            metric_logger.loss_in.global_avg,
            metric_logger.loss_out.global_avg,
            metric_logger.psnr_in.global_avg,
            metric_logger.psnr_out.global_avg,
        )
    )

    if logger:
        if P.log_method == "step":
            for i in range(P.num_submodules):
                for j in range(P.num_submodules):
                    logger.writer.add_scalar(
                        f"eval_loss_in_step{i:02}/loss_patch{j:02}",
                        loss_in_log[i][j].mean().item(),
                        steps,
                    )
                    logger.writer.add_scalar(
                        f"eval_psnr_in_step{i:02}/psnr_patch{j:02}",
                        psnr(loss_in_log[i][j]).mean().item(),
                        steps,
                    )
                if i > 0:
                    logger.writer.add_scalar(
                        f"eval_loss_in_step{i:02}",
                        loss_in_log[i][:i].mean().item(),
                        steps,
                    )
                    logger.writer.add_scalar(
                        f"eval_psnr_in_step{i:02}",
                        psnr(loss_in_log[i][:i]).mean().item(),
                        steps,
                    )
        elif P.log_method == "patch":
            for i in range(P.num_submodules):
                for j in range(P.num_submodules):
                    logger.writer.add_scalar(
                        f"eval_loss_in_patch{i:02}/loss_step{j:02}",
                        loss_in_log[j][i].mean().item(),
                        steps,
                    )
                    logger.writer.add_scalar(
                        f"eval_psnr_in_patch{i:02}/psnr_step{j:02}",
                        psnr(loss_in_log[j][i]).mean().item(),
                        steps,
                    )

        logger.scalar_summary("eval/inner_lr", P.trained_inner_lr, steps)
        logger.scalar_summary("eval/loss_in", metric_logger.loss_in.global_avg, steps)
        logger.scalar_summary("eval/loss_out", metric_logger.loss_out.global_avg, steps)
        logger.scalar_summary("eval/psnr_in", metric_logger.psnr_in.global_avg, steps)
        logger.scalar_summary("eval/psnr_out", metric_logger.psnr_out.global_avg, steps)
        if P.data_type == "video":
            logger.video_summary(
                "eval/vids_in", res_in, metric_logger.eval_context, steps
            )
            logger.video_summary(
                "eval/vids_out", res_out, metric_logger.eval_context, steps
            )

    wrapper.train(mode)
    return metric_logger.psnr_out.global_avg
