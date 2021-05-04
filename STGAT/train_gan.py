import argparse
import gc
import logging
import os
import sys
import time
import yaml
import random
import numpy as np

from collections import defaultdict

import torch
import torch.nn as nn
import torch.optim as optim

from data.loader import data_loader

from models import TrajectoryGenerator, TrajectoryDiscriminator
from utils import (
    displacement_error,
    final_displacement_error,
    get_dset_path,
    int_tuple,
    l2_loss,
    gan_d_loss,
    gan_g_loss,
    relative_to_abs,
    bool_flag,
    get_total_norm,
)

# from sgan.losses import gan_g_loss, gan_d_loss, l2_loss
# from sgan.losses import displacement_error, final_displacement_error

# from sgan.models import TrajectoryGenerator, TrajectoryDiscriminator
# from sgan.utils import int_tuple, bool_flag, get_total_norm
# from sgan.utils import relative_to_abs, get_dset_path

torch.backends.cudnn.benchmark = True

parser = argparse.ArgumentParser()
FORMAT = "[%(levelname)s: %(filename)s: %(lineno)4d]: %(message)s"
logging.basicConfig(level=logging.INFO, format=FORMAT, stream=sys.stdout)

# logger = logging.getLogger(__name__)
# set_logger(os.path.join(args.output_dir, f"train_{args.dataset_name}.log"))


# Dataset options
parser.add_argument("--dataset_name", default="zara1", type=str)
parser.add_argument("--delim", default=" ")
parser.add_argument("--loader_num_workers", default=4, type=int)
parser.add_argument("--obs_len", default=8, type=int)
parser.add_argument("--pred_len", default=8, type=int)
parser.add_argument("--skip", default=1, type=int)

# Optimization
parser.add_argument("--batch_size", default=64, type=int)
parser.add_argument("--num_iterations", default=10000, type=int)
parser.add_argument("--num_epochs", default=200, type=int)

# Model Options
parser.add_argument("--noise_dim", default=(16,), type=int_tuple)
parser.add_argument("--noise_type", default="gaussian")

parser.add_argument(
    "--traj_lstm_input_size", type=int, default=2, help="traj_lstm_input_size"
)
parser.add_argument("--traj_lstm_hidden_size", default=32, type=int)

parser.add_argument(
    "--heads", type=str, default="4,1", help="Heads in each layer, splitted with comma"
)
parser.add_argument(
    "--hidden-units",
    type=str,
    default="16",
    help="Hidden units in each hidden layer, splitted with comma",
)
parser.add_argument(
    "--graph_network_out_dims",
    type=int,
    default=32,
    help="dims of every node after through GAT module",
)
parser.add_argument("--graph_lstm_hidden_size", default=32, type=int)

parser.add_argument(
    "--dropout", type=float, default=0, help="Dropout rate (1 - keep probability)."
)
parser.add_argument(
    "--alpha", type=float, default=0.2, help="Alpha for the leaky_relu."
)


# parser.add_argument("--embedding_dim", default=64, type=int)
# parser.add_argument("--num_layers", default=1, type=int)
# parser.add_argument("--dropout", default=0, type=float)
parser.add_argument("--batch_norm", default=0, type=bool_flag)
parser.add_argument("--mlp_dim", default=256, type=int)

# Generator Options
# parser.add_argument("--encoder_h_dim_g", default=64, type=int)
# parser.add_argument("--decoder_h_dim_g", default=128, type=int)
# parser.add_argument("--noise_dim", default=None, type=int_tuple)
# parser.add_argument("--noise_type", default="gaussian")
# parser.add_argument("--noise_mix_type", default="ped")
parser.add_argument("--clipping_threshold_g", default=0, type=float)
parser.add_argument("--g_learning_rate", default=5e-4, type=float)
parser.add_argument("--g_steps", default=1, type=int)

# Pooling Options
# parser.add_argument("--pooling_type", default="pool_net")
# parser.add_argument("--pool_every_timestep", default=1, type=bool_flag)

# Pool Net Option
# parser.add_argument("--bottleneck_dim", default=1024, type=int)

# Social Pooling Options
# parser.add_argument("--neighborhood_size", default=2.0, type=float)
# parser.add_argument("--grid_size", default=8, type=int)

# Discriminator Options
parser.add_argument("--d_type", default="local", type=str)
parser.add_argument("--encoder_h_dim_d", default=64, type=int)
parser.add_argument("--d_learning_rate", default=5e-4, type=float)
parser.add_argument("--d_steps", default=2, type=int)
parser.add_argument("--clipping_threshold_d", default=0, type=float)

# Loss Options
parser.add_argument("--l2_loss_weight", default=0, type=float)
parser.add_argument("--best_k", default=1, type=int)

# Output
parser.add_argument("--output_dir", default=os.getcwd())
parser.add_argument("--print_every", default=5, type=int)
parser.add_argument("--checkpoint_every", default=100, type=int)
parser.add_argument("--checkpoint_name", default="checkpoint")
parser.add_argument("--checkpoint_start_from", default=None)
parser.add_argument("--restore_from_checkpoint", default=1, type=int)
parser.add_argument("--num_samples_check", default=5000, type=int)

# Misc
parser.add_argument("--use_gpu", default=0, type=int)
parser.add_argument("--timing", default=0, type=int)
parser.add_argument("--gpu_num", default="0", type=str)
parser.add_argument("--seed", type=int, default=72, help="Random seed.")


def init_weights(m):
    classname = m.__class__.__name__
    if classname.find("Linear") != -1:
        nn.init.kaiming_normal_(m.weight)


def get_dtypes(args):
    long_dtype = torch.LongTensor
    float_dtype = torch.FloatTensor
    if args.use_gpu == 1:
        long_dtype = torch.cuda.LongTensor
        float_dtype = torch.cuda.FloatTensor
    return long_dtype, float_dtype


def set_logger(log_path, logger):
    """Set the logger to log info in terminal and file `log_path`.

    In general, it is useful to have a logger so that every output to the terminal is saved
    in a permanent file. Here we save it to `model_dir/train.log`.

    Example:
    ```
    logging.info("Starting training...")
    ```

    Args:
        log_path: (string) where to log
    """
    # logger = logging.getLogger()

    if not logger.handlers:
        # Logging to a file
        file_handler = logging.FileHandler(log_path)
        file_handler.setFormatter(logging.Formatter(FORMAT))
        logger.addHandler(file_handler)

        # Logging to console
        # stream_handler = logging.StreamHandler()
        # stream_handler.setFormatter(logging.Formatter(FORMAT))
        # logger.addHandler(stream_handler)


def main(args):
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_num
    train_path = get_dset_path(args.dataset_name, "train")
    val_path = get_dset_path(args.dataset_name, "val")

    long_dtype, float_dtype = get_dtypes(args)

    logger.info("Initializing train dataset")
    train_dset, train_loader = data_loader(args, train_path)
    logger.info("Initializing val dataset")
    _, val_loader = data_loader(args, val_path)

    iterations_per_epoch = len(train_dset) / args.batch_size / args.d_steps
    if args.num_epochs:
        args.num_iterations = int(iterations_per_epoch * args.num_epochs)

    logger.info("There are {} iterations per epoch".format(iterations_per_epoch))

    n_units = (
        [args.traj_lstm_hidden_size]
        + [int(x) for x in args.hidden_units.strip().split(",")]
        + [args.graph_lstm_hidden_size]
    )
    n_heads = [int(x) for x in args.heads.strip().split(",")]

    generator = TrajectoryGenerator(
        obs_len=args.obs_len,
        pred_len=args.pred_len,
        traj_lstm_input_size=args.traj_lstm_input_size,
        traj_lstm_hidden_size=args.traj_lstm_hidden_size,
        n_units=n_units,
        n_heads=n_heads,
        graph_network_out_dims=args.graph_network_out_dims,
        dropout=args.dropout,
        alpha=args.alpha,
        graph_lstm_hidden_size=args.graph_lstm_hidden_size,
        noise_dim=args.noise_dim,
        noise_type=args.noise_type,
    )

    generator.apply(init_weights)
    generator.type(float_dtype).train()
    generator.cuda()

    logger.info("Here is the generator:")
    logger.info(generator)

    discriminator = TrajectoryDiscriminator(
        obs_len=args.obs_len,
        pred_len=args.pred_len,
        traj_lstm_input_size=args.traj_lstm_input_size,
        traj_lstm_hidden_size=args.traj_lstm_hidden_size,
        n_units=n_units,
        n_heads=n_heads,
        graph_network_out_dims=args.graph_network_out_dims,
        graph_lstm_hidden_size=args.graph_lstm_hidden_size,
        mlp_dim=args.mlp_dim,
        noise_dim=args.noise_dim,
        noise_type=args.noise_type,
        batch_norm=args.batch_norm,
        dropout=args.dropout,
        alpha=args.alpha,
    )

    discriminator.apply(init_weights)
    discriminator.type(float_dtype).train()
    discriminator.cuda()

    logger.info("Here is the discriminator:")
    logger.info(discriminator)

    g_loss_fn = gan_g_loss
    d_loss_fn = gan_d_loss

    # optimizer_g = optim.Adam(generator.parameters(), lr=args.g_learning_rate)
    # optimizer_d = optim.Adam(discriminator.parameters(), lr=args.d_learning_rate)

    optimizer_g = optim.Adam(
        [
            {"params": generator.encoder.traj_lstm_model.parameters(), "lr": 1e-2},
            # {"params": model.traj_hidden2pos.parameters()},
            {"params": generator.encoder.gatencoder.parameters(), "lr": 3e-2},
            {"params": generator.encoder.graph_lstm_model.parameters(), "lr": 1e-2},
            # {"params": model.traj_gat_hidden2pos.parameters()},
            {"params": generator.decoder.pred_lstm_model.parameters()},
            {"params": generator.decoder.pred_hidden2pos.parameters()},
        ],
        lr=args.g_learning_rate,
    )

    optimizer_d = optim.Adam(
        [
            {"params": discriminator.encoder.traj_lstm_model.parameters(), "lr": 1e-2},
            # {"params": model.traj_hidden2pos.parameters()},
            {"params": discriminator.encoder.gatencoder.parameters(), "lr": 3e-2},
            {"params": discriminator.encoder.graph_lstm_model.parameters(), "lr": 1e-2},
            # {"params": model.traj_gat_hidden2pos.parameters()},
            {"params": discriminator.real_classifier.parameters()},
        ],
        lr=args.d_learning_rate,
    )

    # Maybe restore from checkpoint
    restore_path = None
    if args.checkpoint_start_from is not None:
        restore_path = args.checkpoint_start_from
    elif args.restore_from_checkpoint == 1:
        restore_path = os.path.join(
            args.output_dir, "%s_with_model.pt" % args.checkpoint_name
        )

    if restore_path is not None and os.path.isfile(restore_path):
        logger.info("Restoring from checkpoint {}".format(restore_path))
        checkpoint = torch.load(restore_path)
        generator.load_state_dict(checkpoint["g_state"])
        discriminator.load_state_dict(checkpoint["d_state"])
        optimizer_g.load_state_dict(checkpoint["g_optim_state"])
        optimizer_d.load_state_dict(checkpoint["d_optim_state"])
        t = checkpoint["counters"]["t"]
        epoch = checkpoint["counters"]["epoch"]
        checkpoint["restore_ts"].append(t)
    else:
        # Starting from scratch, so initialize checkpoint data structure
        t, epoch = 0, 0
        checkpoint = {
            "args": args.__dict__,
            "G_losses": defaultdict(list),
            "D_losses": defaultdict(list),
            "losses_ts": [],
            "metrics_val": defaultdict(list),
            "metrics_train": defaultdict(list),
            "sample_ts": [],
            "restore_ts": [],
            "norm_g": [],
            "norm_d": [],
            "counters": {
                "t": None,
                "epoch": None,
            },
            "g_state": None,
            "g_optim_state": None,
            "d_state": None,
            "d_optim_state": None,
            "g_best_state": None,
            "d_best_state": None,
            "best_t": None,
            "g_best_nl_state": None,
            "d_best_state_nl": None,
            "best_t_nl": None,
        }
    t0 = None
    while t < args.num_iterations:
        gc.collect()
        d_steps_left = args.d_steps
        g_steps_left = args.g_steps
        epoch += 1
        logger.info("Starting epoch {}".format(epoch))
        for batch in train_loader:
            if args.timing == 1:
                torch.cuda.synchronize()
                t1 = time.time()

            # Decide whether to use the batch for stepping on discriminator or
            # generator; an iteration consists of args.d_steps steps on the
            # discriminator followed by args.g_steps steps on the generator.
            if d_steps_left > 0:
                step_type = "d"
                losses_d = discriminator_step(
                    args, batch, generator, discriminator, d_loss_fn, optimizer_d
                )
                checkpoint["norm_d"].append(get_total_norm(discriminator.parameters()))
                d_steps_left -= 1
            elif g_steps_left > 0:
                step_type = "g"
                losses_g = generator_step(
                    args, batch, generator, discriminator, g_loss_fn, optimizer_g
                )
                checkpoint["norm_g"].append(get_total_norm(generator.parameters()))
                g_steps_left -= 1

            if args.timing == 1:
                torch.cuda.synchronize()
                t2 = time.time()
                logger.info("{} step took {}".format(step_type, t2 - t1))

            # Skip the rest if we are not at the end of an iteration
            if d_steps_left > 0 or g_steps_left > 0:
                continue

            if args.timing == 1:
                if t0 is not None:
                    logger.info("Interation {} took {}".format(t - 1, time.time() - t0))
                t0 = time.time()

            # Maybe save loss
            if t % args.print_every == 0:
                logger.info("t = {} / {}".format(t + 1, args.num_iterations))
                for k, v in sorted(losses_d.items()):
                    logger.info("  [D] {}: {:.3f}".format(k, v))
                    checkpoint["D_losses"][k].append(v)
                for k, v in sorted(losses_g.items()):
                    logger.info("  [G] {}: {:.3f}".format(k, v))
                    checkpoint["G_losses"][k].append(v)
                checkpoint["losses_ts"].append(t)

            # Maybe save a checkpoint
            if t > 0 and t % args.checkpoint_every == 0:
                checkpoint["counters"]["t"] = t
                checkpoint["counters"]["epoch"] = epoch
                checkpoint["sample_ts"].append(t)

                # Check stats on the validation set
                logger.info("Checking stats on val ...")
                metrics_val = check_accuracy(
                    args, val_loader, generator, discriminator, d_loss_fn
                )
                logger.info("Checking stats on train ...")
                metrics_train = check_accuracy(
                    args, train_loader, generator, discriminator, d_loss_fn, limit=True
                )

                for k, v in sorted(metrics_val.items()):
                    logger.info("  [val] {}: {:.3f}".format(k, v))
                    checkpoint["metrics_val"][k].append(v)
                for k, v in sorted(metrics_train.items()):
                    logger.info("  [train] {}: {:.3f}".format(k, v))
                    checkpoint["metrics_train"][k].append(v)

                min_ade = min(checkpoint["metrics_val"]["ade"])
                min_ade_nl = min(checkpoint["metrics_val"]["ade_nl"])

                if metrics_val["ade"] == min_ade:
                    logger.info("New low for avg_disp_error")
                    checkpoint["best_t"] = t
                    checkpoint["g_best_state"] = generator.state_dict()
                    checkpoint["d_best_state"] = discriminator.state_dict()

                if metrics_val["ade_nl"] == min_ade_nl:
                    logger.info("New low for avg_disp_error_nl")
                    checkpoint["best_t_nl"] = t
                    checkpoint["g_best_nl_state"] = generator.state_dict()
                    checkpoint["d_best_nl_state"] = discriminator.state_dict()

                # Save another checkpoint with model weights and
                # optimizer state
                checkpoint["g_state"] = generator.state_dict()
                checkpoint["g_optim_state"] = optimizer_g.state_dict()
                checkpoint["d_state"] = discriminator.state_dict()
                checkpoint["d_optim_state"] = optimizer_d.state_dict()
                checkpoint_path = os.path.join(
                    args.output_dir, "%s_with_model.pt" % args.checkpoint_name
                )
                logger.info("Saving checkpoint to {}".format(checkpoint_path))
                torch.save(checkpoint, checkpoint_path)
                logger.info("Done.")

                # Save a checkpoint with no model weights by making a shallow
                # copy of the checkpoint excluding some items
                checkpoint_path = os.path.join(
                    args.output_dir, "%s_no_model.pt" % args.checkpoint_name
                )
                logger.info("Saving checkpoint to {}".format(checkpoint_path))
                key_blacklist = [
                    "g_state",
                    "d_state",
                    "g_best_state",
                    "g_best_nl_state",
                    "g_optim_state",
                    "d_optim_state",
                    "d_best_state",
                    "d_best_nl_state",
                ]
                small_checkpoint = {}
                for k, v in checkpoint.items():
                    if k not in key_blacklist:
                        small_checkpoint[k] = v
                torch.save(small_checkpoint, checkpoint_path)
                logger.info("Done.")

            t += 1
            d_steps_left = args.d_steps
            g_steps_left = args.g_steps
            if t >= args.num_iterations:
                break


def discriminator_step(args, batch, generator, discriminator, d_loss_fn, optimizer_d):
    batch = [tensor.cuda() for tensor in batch]
    (
        obs_traj,
        pred_traj_gt,
        obs_traj_rel,
        pred_traj_gt_rel,
        non_linear_ped,
        loss_mask,
        seq_start_end,
        # cls_labels,
    ) = batch  # TEST: cGAN
    losses = {}
    loss = torch.zeros(1).to(pred_traj_gt)

    model_input = torch.cat((obs_traj_rel, pred_traj_gt_rel), dim=0)

    # generator_out = generator(obs_traj, obs_traj_rel, cls_labels, seq_start_end)
    generator_out = generator(model_input, obs_traj_rel, seq_start_end)

    pred_traj_fake_rel = generator_out
    pred_traj_fake = relative_to_abs(pred_traj_fake_rel, obs_traj[-1])

    traj_real = torch.cat([obs_traj, pred_traj_gt], dim=0)
    traj_real_rel = torch.cat([obs_traj_rel, pred_traj_gt_rel], dim=0)
    traj_fake = torch.cat([obs_traj, pred_traj_fake], dim=0)
    traj_fake_rel = torch.cat([obs_traj_rel, pred_traj_fake_rel], dim=0)

    scores_fake = discriminator(traj_fake, traj_fake_rel, seq_start_end)
    scores_real = discriminator(traj_real, traj_real_rel, seq_start_end)

    # Compute loss with optional gradient penalty
    data_loss = d_loss_fn(scores_real, scores_fake)
    losses["D_data_loss"] = data_loss.item()
    loss += data_loss
    losses["D_total_loss"] = loss.item()

    optimizer_d.zero_grad()
    loss.backward()
    if args.clipping_threshold_d > 0:
        nn.utils.clip_grad_norm_(discriminator.parameters(), args.clipping_threshold_d)
    optimizer_d.step()

    return losses


def generator_step(args, batch, generator, discriminator, g_loss_fn, optimizer_g):
    batch = [tensor.cuda() for tensor in batch]
    (
        obs_traj,
        pred_traj_gt,
        obs_traj_rel,
        pred_traj_gt_rel,
        non_linear_ped,
        loss_mask,
        seq_start_end,
        # cls_labels,
    ) = batch  # TEST: cGAN
    losses = {}
    loss = torch.zeros(1).to(pred_traj_gt)
    g_l2_loss_rel = []

    loss_mask = loss_mask[:, args.obs_len :]

    model_input = torch.cat((obs_traj_rel, pred_traj_gt_rel), dim=0)

    for _ in range(args.best_k):
        
        generator_out = generator(model_input, obs_traj_rel, seq_start_end)  # TEST: cGAN

        pred_traj_fake_rel = generator_out
        pred_traj_fake = relative_to_abs(pred_traj_fake_rel, obs_traj[-1])

        if args.l2_loss_weight > 0:
            g_l2_loss_rel.append(
                args.l2_loss_weight
                * l2_loss(pred_traj_fake_rel, pred_traj_gt_rel, loss_mask, mode="raw")
            )

    g_l2_loss_sum_rel = torch.zeros(1).to(pred_traj_gt)
    if args.l2_loss_weight > 0:
        g_l2_loss_rel = torch.stack(g_l2_loss_rel, dim=1)
        for start, end in seq_start_end.data:
            _g_l2_loss_rel = g_l2_loss_rel[start:end]
            _g_l2_loss_rel = torch.sum(_g_l2_loss_rel, dim=0)
            _g_l2_loss_rel = torch.min(_g_l2_loss_rel) / torch.sum(loss_mask[start:end])
            g_l2_loss_sum_rel += _g_l2_loss_rel
        losses["G_l2_loss_rel"] = g_l2_loss_sum_rel.item()
        loss += g_l2_loss_sum_rel

    traj_fake = torch.cat([obs_traj, pred_traj_fake], dim=0)
    traj_fake_rel = torch.cat([obs_traj_rel, pred_traj_fake_rel], dim=0)

    scores_fake = discriminator(traj_fake, traj_fake_rel, seq_start_end)  # TEST: cGAN
    discriminator_loss = g_loss_fn(scores_fake)

    loss += discriminator_loss
    losses["G_discriminator_loss"] = discriminator_loss.item()
    losses["G_total_loss"] = loss.item()

    optimizer_g.zero_grad()
    loss.backward()
    if args.clipping_threshold_g > 0:
        nn.utils.clip_grad_norm_(generator.parameters(), args.clipping_threshold_g)
    optimizer_g.step()

    return losses


def check_accuracy(args, loader, generator, discriminator, d_loss_fn, limit=False):
    d_losses = []
    metrics = {}
    g_l2_losses_abs, g_l2_losses_rel = ([],) * 2
    disp_error, disp_error_l, disp_error_nl = ([],) * 3
    f_disp_error, f_disp_error_l, f_disp_error_nl = ([],) * 3
    total_traj, total_traj_l, total_traj_nl = 0, 0, 0
    loss_mask_sum = 0
    generator.eval()
    with torch.no_grad():
        for batch in loader:
            batch = [tensor.cuda() for tensor in batch]
            (
                obs_traj,
                pred_traj_gt,
                obs_traj_rel,
                pred_traj_gt_rel,
                non_linear_ped,
                loss_mask,
                seq_start_end,
                # cls_labels,  # TEST: cGAN
            ) = batch
            linear_ped = 1 - non_linear_ped
            loss_mask = loss_mask[:, args.obs_len :]

            pred_traj_fake_rel = generator(
                obs_traj, obs_traj_rel, seq_start_end
            )
            pred_traj_fake = relative_to_abs(pred_traj_fake_rel, obs_traj[-1])

            g_l2_loss_abs, g_l2_loss_rel = cal_l2_losses(
                pred_traj_gt,
                pred_traj_gt_rel,
                pred_traj_fake,
                pred_traj_fake_rel,
                loss_mask,
            )
            ade, ade_l, ade_nl = cal_ade(
                pred_traj_gt, pred_traj_fake, linear_ped, non_linear_ped
            )

            fde, fde_l, fde_nl = cal_fde(
                pred_traj_gt, pred_traj_fake, linear_ped, non_linear_ped
            )

            traj_real = torch.cat([obs_traj, pred_traj_gt], dim=0)
            traj_real_rel = torch.cat([obs_traj_rel, pred_traj_gt_rel], dim=0)
            traj_fake = torch.cat([obs_traj, pred_traj_fake], dim=0)
            traj_fake_rel = torch.cat([obs_traj_rel, pred_traj_fake_rel], dim=0)

            scores_fake = discriminator(
                traj_fake, traj_fake_rel, seq_start_end
            )
            scores_real = discriminator(
                traj_real, traj_real_rel, seq_start_end
            )

            d_loss = d_loss_fn(scores_real, scores_fake)
            d_losses.append(d_loss.item())

            g_l2_losses_abs.append(g_l2_loss_abs.item())
            g_l2_losses_rel.append(g_l2_loss_rel.item())
            disp_error.append(ade.item())
            disp_error_l.append(ade_l.item())
            disp_error_nl.append(ade_nl.item())
            f_disp_error.append(fde.item())
            f_disp_error_l.append(fde_l.item())
            f_disp_error_nl.append(fde_nl.item())

            loss_mask_sum += torch.numel(loss_mask.data)
            total_traj += pred_traj_gt.size(1)
            total_traj_l += torch.sum(linear_ped).item()
            total_traj_nl += torch.sum(non_linear_ped).item()
            if limit and total_traj >= args.num_samples_check:
                break

    metrics["d_loss"] = sum(d_losses) / len(d_losses)
    metrics["g_l2_loss_abs"] = sum(g_l2_losses_abs) / loss_mask_sum
    metrics["g_l2_loss_rel"] = sum(g_l2_losses_rel) / loss_mask_sum

    metrics["ade"] = sum(disp_error) / (total_traj * args.pred_len)
    metrics["fde"] = sum(f_disp_error) / total_traj
    if total_traj_l != 0:
        metrics["ade_l"] = sum(disp_error_l) / (total_traj_l * args.pred_len)
        metrics["fde_l"] = sum(f_disp_error_l) / total_traj_l
    else:
        metrics["ade_l"] = 0
        metrics["fde_l"] = 0
    if total_traj_nl != 0:
        metrics["ade_nl"] = sum(disp_error_nl) / (total_traj_nl * args.pred_len)
        metrics["fde_nl"] = sum(f_disp_error_nl) / total_traj_nl
    else:
        metrics["ade_nl"] = 0
        metrics["fde_nl"] = 0

    generator.train()
    return metrics


def cal_l2_losses(
    pred_traj_gt, pred_traj_gt_rel, pred_traj_fake, pred_traj_fake_rel, loss_mask
):
    g_l2_loss_abs = l2_loss(pred_traj_fake, pred_traj_gt, loss_mask, mode="sum")
    g_l2_loss_rel = l2_loss(pred_traj_fake_rel, pred_traj_gt_rel, loss_mask, mode="sum")
    return g_l2_loss_abs, g_l2_loss_rel


def cal_ade(pred_traj_gt, pred_traj_fake, linear_ped, non_linear_ped):
    ade = displacement_error(pred_traj_fake, pred_traj_gt)
    ade_l = displacement_error(pred_traj_fake, pred_traj_gt, linear_ped)
    ade_nl = displacement_error(pred_traj_fake, pred_traj_gt, non_linear_ped)
    return ade, ade_l, ade_nl


def cal_fde(pred_traj_gt, pred_traj_fake, linear_ped, non_linear_ped):
    fde = final_displacement_error(pred_traj_fake[-1], pred_traj_gt[-1])
    fde_l = final_displacement_error(pred_traj_fake[-1], pred_traj_gt[-1], linear_ped)
    fde_nl = final_displacement_error(
        pred_traj_fake[-1], pred_traj_gt[-1], non_linear_ped
    )
    return fde, fde_l, fde_nl


if __name__ == "__main__":
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True

    # keep track of console outputs and experiment settings
    os.makedirs(args.output_dir, exist_ok=True)
    config_file = open(
        os.path.join(args.output_dir, f"config_{args.dataset_name}.yaml"), "w"
    )
    yaml.dump(args, config_file)
    logger = logging.getLogger(__name__)
    set_logger(os.path.join(args.output_dir, f"train_{args.dataset_name}.log"), logger)

    main(args)
