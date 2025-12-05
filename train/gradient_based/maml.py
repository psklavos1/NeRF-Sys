import time
import torch

from train.gradient_based.__init__ import inner_adapt, divide_loss
from utils import psnr, get_meta_batch


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def log_theta(msg, params):
    for name, param in params:
        print(f"{name}: {param.data.mean()}")
    print(msg)


def train_step_img(
    P, steps, wrapper, optimizer, task_data, metric_logger, logger, inner_lr
):
    """
    Executes a single meta-training step for image tasks.

    Args:
        P: Namespace or configuration object containing training parameters.
        steps: Current training step.
        wrapper: MetaWrapper model instance.
        optimizer: Optimizer for meta-parameters.
        task_data: Batch of image task data.
        metric_logger: Metric logger instance.
        logger: Logger instance for recording scalar values and images.
        inner_lr: Current inner-loop learning rate (can be learned).
    """
    stime = time.time()
    wrapper.train()

    batch_size, episode_batch = get_meta_batch(
        P, task_data
    )  # int, tensor.Size([B C H W])

    # Run inner loop
    wrapper.support = True
    wrapper.sample(
        sample_type="random"
    )  # subsample the whole grid to create support set

    # log_theta_0(msg="Before inner adapt", params=wrapper.decoder.named_parameters())

    # loss_in: (step * iter, b, c, h//n, w//n), loss_in_log: (step, t, b, c, h//n, w//n), res_in: (step, b, c, h//n, w//n)
    params, loss_in, loss_in_log, res_in, grad_in = inner_adapt(
        P,
        wrapper,
        episode_batch,
        inner_lr,
        P.num_submodules,
        first_order=P.algo == "fomaml",
        order=P.order,
    )
    loss_in = loss_in.view(P.num_submodules, P.inner_iter, batch_size, -1).mean(
        dim=-1
    )  # (step, iter, b)

    loss_in = loss_in[-1][-1]  # (b)

    loss_in_log = loss_in_log.view(
        P.num_submodules, P.num_submodules, batch_size, -1
    ).mean(dim=-1)

    """ outer loss aggregate """
    # log_theta_0(msg="Before meta-update", params=wrapper.decoder.named_parameters())
    wrapper.support = False
    loss_out, res_out = wrapper(episode_batch, params=params)  # (b, c, h, w)

    B, C, H, W = loss_out.shape
    flat_loss = loss_out.view(B, C, H * W)
    q = wrapper.query_indices  # LongTensor of query pixel indices
    loss_out = flat_loss[:, :, q].mean(dim=(1, 2))  # (B,)
    loss = loss_out.mean() * B
    # loss_out = loss_out.view(batch_size, -1).mean(dim=1)  # (b)
    # loss = loss_out.mean() * batch_size

    """ outer gradient step """
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(wrapper.decoder.parameters(), 1.0)
    before = {name: p.data.clone() for name, p in wrapper.decoder.named_parameters()}
    optimizer.step()
    torch.cuda.synchronize()
    P.trained_inner_lr = inner_lr.item()

    for name, p in wrapper.decoder.named_parameters():
        delta = (p.data - before[name]).abs().mean().item()
        print(f"{name} change: {delta:.6f}")
    print("After update Change")
    # log_theta_0(msg="After meta-update", params=wrapper.decoder.named_parameters())
    """ track stat """
    metric_logger.meters["batch_time"].update(time.time() - stime, n=batch_size)
    metric_logger.meters["train_context"].update(
        episode_batch, n=batch_size
    )  # TODO if  error it may need []
    metric_logger.meters["loss_in"].update(loss_in.mean().item(), n=batch_size)
    metric_logger.meters["psnr_in"].update(psnr(loss_in).mean().item(), n=batch_size)
    metric_logger.meters["loss_out"].update(loss_out.mean().item(), n=batch_size)
    metric_logger.meters["psnr_out"].update(psnr(loss_out).mean().item(), n=batch_size)
    metric_logger.synchronize_between_processes()

    if steps % P.print_step == 0:
        logger.log_dirname(f"Step {steps}")
        if P.log_method == "step":
            for i in range(P.num_submodules):
                for j in range(P.num_submodules):
                    logger.writer.add_scalar(
                        f"train_loss_in_step{i:02}/loss_patch{j:02}",
                        loss_in_log[i][j].mean().item(),
                        steps,
                    )
                    logger.writer.add_scalar(
                        f"train_psnr_in_step{i:02}/psnr_patch{j:02}",
                        psnr(loss_in_log[i][j]).mean().item(),
                        steps,
                    )
                if i > 0:
                    logger.writer.add_scalar(
                        f"train_loss_in_step{i:02}",
                        loss_in_log[i][:i].mean().item(),
                        steps,
                    )
                    logger.writer.add_scalar(
                        f"train_psnr_in_step{i:02}",
                        psnr(loss_in_log[i][:i]).mean().item(),
                        steps,
                    )
        elif P.log_method == "patch":
            for i in range(P.num_submodules):
                for j in range(P.num_submodules):
                    logger.writer.add_scalar(
                        f"train_loss_in_patch{i:02}/loss_step{j:02}",
                        loss_in_log[j][i].mean().item(),
                        steps,
                    )
                    logger.writer.add_scalar(
                        f"train_psnr_in_patch{i:02}/psnr_step{j:02}",
                        psnr(loss_in_log[j][i]).mean().item(),
                        steps,
                    )
                if i > 0:
                    logger.writer.add_scalar(
                        f"train_loss_in_step{i:02}",
                        loss_in_log[i][:i].mean().item(),
                        steps,
                    )
                    logger.writer.add_scalar(
                        f"train_psnr_in_step{i:02}",
                        psnr(loss_in_log[i][:i]).mean().item(),
                        steps,
                    )

        logger.scalar_summary("train/inner_lr", inner_lr.item(), steps)
        logger.scalar_summary("train/loss_in", loss_in.mean().item(), steps)
        logger.scalar_summary("train/loss_out", loss_out.mean().item(), steps)
        logger.scalar_summary("train/psnr_in", psnr(loss_in).mean().item(), steps)
        logger.scalar_summary("train/psnr_out", psnr(loss_out).mean().item(), steps)
        logger.scalar_summary("train/batch_time", metric_logger.batch_time.value, steps)
        # if P.data_type == "img":
        # logger.image_summary(
        #     "train/imgs_in", res_in, metric_logger.train_context, steps
        # )
        # logger.image_summary(
        #     "train/imgs_out", res_out, metric_logger.train_context, steps
        # )

        logger.log(
            "[TRAIN] [Step %3d] [Time %.3f] [Data %.3f] "
            "[LossIn %f] [LossOut %f] [PSNRIn %.3f] [PSNROut %.3f]"
            % (
                steps,
                metric_logger.batch_time.global_avg,
                metric_logger.data_time.global_avg,
                loss_in.mean().item(),
                loss_out.mean().item(),
                psnr(loss_in).mean().item(),
                psnr(loss_out).mean().item(),
            )
        )

        metric_logger.reset()


def train_step_video(
    P, steps, wrapper, optimizer, task_data, metric_logger, logger, inner_lr
):
    """
    Executes a single meta-training step for video tasks.

    Args:
        P: Namespace or configuration object containing training parameters.
        steps: Current training step.
        wrapper: MetaWrapper model instance.
        optimizer: Optimizer for meta-parameters.
        task_data: Batch of video task data.
        metric_logger: Metric logger instance.
        logger: Logger instance for recording scalar values and videos.
        inner_lr: Current inner-loop learning rate (can be learned).
    """
    stime = time.time()
    wrapper.train()

    # TODO: Why batches of images? Why random selection of regions
    # TODO: Not random! it needs to be from specific region dicated by the step inner!
    batch_size, context = get_meta_batch(P, task_data)

    wrapper.support = True
    params, loss_in, loss_in_log, res_in, grad_in = inner_adapt(
        P,
        wrapper,
        context[0],
        inner_lr,
        P.num_submodules,
        first_order=P.algo == "fomaml",
        order=P.order,
    )

    loss_in = loss_in.view(P.num_submodules, P.inner_iter, batch_size, -1).mean(dim=-1)
    loss_in = loss_in[-1][-1]
    loss_in_log = loss_in_log.view(
        P.num_submodules, P.num_submodules, batch_size, -1
    ).mean(dim=-1)

    wrapper.support = False
    loss_out, res_out = wrapper(context[0], params=params)
    loss_out = loss_out.view(batch_size, -1).mean(dim=-1)
    loss = loss_out.mean()

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(wrapper.decoder.parameters(), 1.0)
    optimizer.step()
    torch.cuda.synchronize()
    P.trained_inner_lr = inner_lr.item()

    metric_logger.meters["batch_time"].update(time.time() - stime, n=batch_size)
    metric_logger.meters["train_context"].update(context[0], n=batch_size)
    metric_logger.synchronize_between_processes()

    if steps % P.print_step == 0:
        logger.log_dirname(f"Step {steps}")

        if P.log_method == "step":
            for i in range(P.num_submodules):
                for j in range(P.num_submodules):
                    logger.writer.add_scalar(
                        f"train_loss_in_step{i:02}/loss_patch{j:02}",
                        loss_in_log[i][j].mean().item(),
                        steps,
                    )
                    logger.writer.add_scalar(
                        f"train_psnr_in_step{i:02}/psnr_patch{j:02}",
                        psnr(loss_in_log[i][j]).mean().item(),
                        steps,
                    )
                if i > 0:
                    logger.writer.add_scalar(
                        f"train_loss_in_step{i:02}",
                        loss_in_log[i][:i].mean().item(),
                        steps,
                    )
                    logger.writer.add_scalar(
                        f"train_psnr_in_step{i:02}",
                        psnr(loss_in_log[i][:i]).mean().item(),
                        steps,
                    )

        elif P.log_method == "patch":
            for i in range(P.num_submodules):
                for j in range(P.num_submodules):
                    logger.writer.add_scalar(
                        f"train_loss_in_patch{i:02}/loss_step{j:02}",
                        loss_in_log[j][i].mean().item(),
                        steps,
                    )
                    logger.writer.add_scalar(
                        f"train_psnr_in_patch{i:02}/psnr_step{j:02}",
                        psnr(loss_in_log[j][i]).mean().item(),
                        steps,
                    )
                if i > 0:
                    logger.writer.add_scalar(
                        f"train_loss_in_step{i:02}",
                        loss_in_log[i][:i].mean().item(),
                        steps,
                    )
                    logger.writer.add_scalar(
                        f"train_psnr_in_step{i:02}",
                        psnr(loss_in_log[i][:i]).mean().item(),
                        steps,
                    )

        logger.scalar_summary("train/inner_lr", inner_lr.item(), steps)
        logger.scalar_summary("train/loss_in", loss_in.mean().item(), steps)
        logger.scalar_summary("train/loss_out", loss_out.mean().item(), steps)
        logger.scalar_summary("train/psnr_in", psnr(loss_in).mean().item(), steps)
        logger.scalar_summary("train/psnr_out", psnr(loss_out).mean().item(), steps)
        logger.scalar_summary("train/batch_time", metric_logger.batch_time.value, steps)

        if P.data_type == "video":
            logger.video_summary(
                "train/vids_in", res_in, metric_logger.train_context, steps
            )
            logger.video_summary(
                "train/vids_out", res_out, metric_logger.train_context, steps
            )

        logger.log(
            "[TRAIN] [Step %3d] [Time %.3f] [Data %.3f] "
            "[LossIn %f] [LossOut %f] [PSNRIn %.3f] [PSNROut %.3f]"
            % (
                steps,
                metric_logger.batch_time.global_avg,
                metric_logger.data_time.global_avg,
                loss_in.mean().item(),
                loss_out.mean().item(),
                psnr(loss_in).mean().item(),
                psnr(loss_out).mean().item(),
            )
        )

        metric_logger.reset()
