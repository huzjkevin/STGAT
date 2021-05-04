import argparse
import logging
import os
import random
import shutil
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

import utils
from data.loader import data_loader
from models import TrajectoryGenerator, TrajectoryDiscriminator
from utils import (
    displacement_error,
    final_displacement_error,
    get_dset_path,
    int_tuple,
    l2_loss,
    relative_to_abs,
    gan_g_loss,
    gan_d_loss,
    bool_flag,
)

parser = argparse.ArgumentParser()

parser.add_argument("--log_dir", default="./", help="Directory containing logging file")
parser.add_argument("--verbose", action="store_true")

parser.add_argument("--dataset_name", default="zara2", type=str)
parser.add_argument("--delim", default="\t")
parser.add_argument("--loader_num_workers", default=4, type=int)
parser.add_argument("--obs_len", default=8, type=int)
parser.add_argument("--pred_len", default=12, type=int)
parser.add_argument("--skip", default=1, type=int)

parser.add_argument("--seed", type=int, default=72, help="Random seed.")
parser.add_argument("--batch_size", default=64, type=int)
parser.add_argument("--num_epochs", default=200, type=int)

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

parser.add_argument("--batch_norm", default=0, type=bool_flag)
parser.add_argument("--mlp_dim", default=512, type=int)

parser.add_argument("--clipping_threshold_g", default=0, type=float)
parser.add_argument("--clipping_threshold_d", default=0, type=float)
parser.add_argument("--g_lr", default=5e-4, type=float)
parser.add_argument("--d_lr", default=5e-4, type=float)

parser.add_argument(
    "--start-epoch",
    default=0,
    type=int,
    metavar="N",
    help="manual epoch number (useful on restarts)",
)

parser.add_argument("--best_k", default=20, type=int)
parser.add_argument("--print_every", default=10, type=int)
parser.add_argument("--use_gpu", default=1, type=int)
parser.add_argument("--gpu_num", default="0", type=str)
parser.add_argument("--val_interval", default=5, type=int)
parser.add_argument("--ckpt_interval", default=10, type=int)

parser.add_argument(
    "--resume",
    default="",
    type=str,
    metavar="PATH",
    help="path to latest checkpoint (default: none)",
)


best_ade = 100


def main(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_num
    train_path = get_dset_path(args.dataset_name, "train")
    val_path = get_dset_path(args.dataset_name, "test")

    logging.info("Initializing train dataset\n")
    train_dset, train_loader = data_loader(args, train_path)
    logging.info("Initializing val dataset\n")
    _, val_loader = data_loader(args, val_path)

    writer = SummaryWriter()

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
    generator.cuda()

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
    discriminator.cuda()

    optimizer_g = optim.Adam(
        [
            {"params": generator.encoder.traj_lstm_model.parameters(), "lr": 1e-3},
            {"params": generator.encoder.gatencoder.parameters(), "lr": 3e-3},
            {"params": generator.encoder.graph_lstm_model.parameters(), "lr": 1e-3},
            {"params": generator.decoder.pred_lstm_model.parameters()},
            {"params": generator.decoder.pred_hidden2pos.parameters()},
        ],
        lr=args.g_lr,
    )

    optimizer_d = optim.Adam(
        [
            {"params": discriminator.encoder.traj_lstm_model.parameters(), "lr": 1e-3},
            {"params": discriminator.encoder.gatencoder.parameters(), "lr": 3e-3},
            {"params": discriminator.encoder.graph_lstm_model.parameters(), "lr": 1e-3},
            {"params": discriminator.real_classifier.parameters()},
        ],
        lr=args.d_lr,
    )
    global best_ade
    if args.resume:
        if os.path.isfile(args.resume):
            logging.info("Restoring from checkpoint {}".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint["epoch"]
            model.load_state_dict(checkpoint["state_dict"])
            logging.info(
                "=> loaded checkpoint '{}' (epoch {})".format(
                    args.resume, checkpoint["epoch"]
                )
            )
        else:
            logging.info("=> no checkpoint found at '{}'".format(args.resume))

    for epoch in range(args.start_epoch, args.num_epochs + 1):
        train_gan(
            args,
            generator,
            discriminator,
            train_loader,
            optimizer_g,
            optimizer_d,
            epoch,
            gan_g_loss,
            gan_d_loss,
            log_writer=writer,
        )

        save_ckpt = epoch % args.ckpt_interval == 0
        val_epoch = epoch % args.val_interval == 0
        ckpt_state = {
            "epoch": epoch + 1,
            "state_g": generator.state_dict(),
            "state_d": discriminator.state_dict(),
            "best_ade": best_ade,
            "optimizer_g": optimizer_g.state_dict(),
            "optimizer_d": optimizer_d.state_dict(),
        }

        if val_epoch:
            ade = validate(args, generator, val_loader, epoch, writer)
            is_best = ade < best_ade
            best_ade = min(ade, best_ade)

            save_checkpoint(
                ckpt_state,
                save_ckpt,
                is_best,
                f"./checkpoint_{args.dataset_name}/checkpoint{epoch}.pth.tar",
            )
        elif save_ckpt:
            save_checkpoint(
                ckpt_state,
                save_ckpt,
                is_best,
                f"./checkpoint_{args.dataset_name}/checkpoint{epoch}.pth.tar",
            )

    writer.close()


def train_generator(
    args,
    batch,
    generator,
    discriminator,
    optimizer_g,
    g_loss_fn,
):
    generator.train()
    discriminator.train()

    (
        obs_traj,
        pred_traj_gt,
        obs_traj_rel,
        pred_traj_gt_rel,
        non_linear_ped,
        loss_mask,
        seq_start_end,
    ) = batch

    optimizer_g.zero_grad()
    loss = torch.zeros(1).to(pred_traj_gt)
    losses = {}
    l2_loss_rel = []
    loss_mask = loss_mask[:, args.obs_len :]

    # generator part
    model_input = torch.cat((obs_traj_rel, pred_traj_gt_rel), dim=0)
    for _ in range(args.best_k):
        pred_traj_fake_rel = generator(model_input, obs_traj, seq_start_end, 0)
        l2_loss_rel.append(
            l2_loss(
                pred_traj_fake_rel,
                model_input[-args.pred_len :],
                loss_mask,
                mode="raw",
            )
        )

    l2_loss_sum_rel = torch.zeros(1).to(pred_traj_gt)
    l2_loss_rel = torch.stack(l2_loss_rel, dim=1)
    for start, end in seq_start_end.data:
        _l2_loss_rel = torch.narrow(l2_loss_rel, 0, start, end - start)
        _l2_loss_rel = torch.sum(_l2_loss_rel, dim=0)  # [20]
        _l2_loss_rel = torch.min(_l2_loss_rel) / (
            (pred_traj_fake_rel.shape[0]) * (end - start)
        )
        l2_loss_sum_rel += _l2_loss_rel

    loss += l2_loss_sum_rel
    losses["G_l2_loss"] = l2_loss_sum_rel.item()
    # discriminator part
    traj_fake_rel = torch.cat([obs_traj_rel, pred_traj_fake_rel], dim=0)
    scores_fake = discriminator(traj_fake_rel, obs_traj, seq_start_end)
    discriminator_loss = g_loss_fn(scores_fake)

    loss += discriminator_loss
    losses["G_discriminator_loss"] = discriminator_loss.item()
    losses["G_total_loss"] = loss.item()

    # losses.update(loss.item(), obs_traj.shape[1])
    loss.backward()
    if args.clipping_threshold_g > 0:
        nn.utils.clip_grad_norm_(generator.parameters(), args.clipping_threshold_g)
    optimizer_g.step()

    return losses


def train_discriminator(
    args,
    batch,
    generator,
    discriminator,
    optimizer_d,
    d_loss_fn,
):
    generator.train()
    discriminator.train()

    (
        obs_traj,
        pred_traj_gt,
        obs_traj_rel,
        pred_traj_gt_rel,
        non_linear_ped,
        loss_mask,
        seq_start_end,
    ) = batch

    optimizer_d.zero_grad()
    loss = torch.zeros(1).to(pred_traj_gt)
    losses = {}
    l2_loss_rel = []
    loss_mask = loss_mask[:, args.obs_len :]

    # generator part
    model_input = torch.cat((obs_traj_rel, pred_traj_gt_rel), dim=0)
    pred_traj_fake_rel = generator(model_input, obs_traj, seq_start_end, 0)

    # discriminator part
    traj_real_rel = torch.cat([obs_traj_rel, pred_traj_gt_rel], dim=0)
    traj_fake_rel = torch.cat([obs_traj_rel, pred_traj_fake_rel], dim=0)

    scores_fake = discriminator(traj_fake_rel, obs_traj, seq_start_end)
    scores_real = discriminator(traj_real_rel, obs_traj, seq_start_end)

    # Compute loss with optional gradient penalty
    data_loss = d_loss_fn(scores_real, scores_fake)
    losses["D_data_loss"] = data_loss.item()
    loss += data_loss
    losses["D_total_loss"] = loss.item()

    loss.backward()
    if args.clipping_threshold_d > 0:
        nn.utils.clip_grad_norm_(discriminator.parameters(), args.clipping_threshold_d)
    optimizer_d.step()

    return losses


def train_gan(
    args,
    generator,
    discriminator,
    train_loader,
    optimizer_g,
    optimizer_d,
    epoch,
    g_loss_fn,
    d_loss_fn,
    log_writer=None,
):
    # losses_g = utils.AverageMeter("Generator loss", ":.6f")
    # losses_d = utils.AverageMeter("Discriminator loss", ":.6f")
    # progress = utils.ProgressMeter(
    #     len(train_loader), [losses], prefix="Epoch: [{}]".format(epoch)
    # )

    tb_dict = {
        "G_l2_loss_epoch": 0,
        "G_discriminator_loss_epoch": 0,
        "G_total_loss_epoch": 0,
        "D_total_loss_epoch": 0,
    }

    if args.verbose:
        logging.info(f"**Train epoch: {epoch} start**")

    for batch_idx, batch in enumerate(train_loader):
        batch = [tensor.cuda() for tensor in batch]
        losses_g = train_generator(
            args, batch, generator, discriminator, optimizer_g, g_loss_fn
        )
        losses_d = train_discriminator(
            args, batch, generator, discriminator, optimizer_d, d_loss_fn
        )

        tb_dict["G_l2_loss_epoch"] += losses_g["G_l2_loss"]
        tb_dict["G_discriminator_loss_epoch"] += losses_g["G_discriminator_loss"]
        tb_dict["G_total_loss_epoch"] += losses_g["G_total_loss"]
        tb_dict["D_total_loss_epoch"] += losses_d["D_total_loss"]

        # print training info to console and log
        if batch_idx % args.print_every == 0 and args.verbose:
            logging.info("  sample batch:")
            for k, v in sorted(losses_g.items()):
                logging.info("  [G] {}: {:.3f}".format(k, v))
                # checkpoint["D_losses"][k].append(v)
            for k, v in sorted(losses_d.items()):
                logging.info("  [D] {}: {:.3f}".format(k, v))
                # checkpoint["G_losses"][k].append(v)
            logging.info("")

    if args.verbose:
        for k, v in sorted(tb_dict.items()):
            log_writer.add_scalar(k, v, epoch)
            logging.info(f"  {k}: {v}")

        logging.info(f"**Train epoch: {epoch} end**\n")
    else:
        logging.info(
            f"Train epoch {epoch}: G_l2_loss_epoch {tb_dict['G_l2_loss_epoch']}"
        )


def train(args, model, train_loader, optimizer, epoch, training_step, writer):
    losses = utils.AverageMeter("Loss", ":.6f")
    progress = utils.ProgressMeter(
        len(train_loader), [losses], prefix="Epoch: [{}]".format(epoch)
    )
    model.train()
    for batch_idx, batch in enumerate(train_loader):
        batch = [tensor.cuda() for tensor in batch]
        (
            obs_traj,
            pred_traj_gt,
            obs_traj_rel,
            pred_traj_gt_rel,
            non_linear_ped,
            loss_mask,
            seq_start_end,
        ) = batch
        optimizer.zero_grad()
        loss = torch.zeros(1).to(pred_traj_gt)
        l2_loss_rel = []
        loss_mask = loss_mask[:, args.obs_len :]

        if training_step == 1 or training_step == 2:
            model_input = obs_traj_rel
            pred_traj_fake_rel = model(
                model_input, obs_traj, seq_start_end, 1, training_step
            )
            l2_loss_rel.append(
                l2_loss(pred_traj_fake_rel, model_input, loss_mask, mode="raw")
            )
        else:
            model_input = torch.cat((obs_traj_rel, pred_traj_gt_rel), dim=0)
            for _ in range(args.best_k):
                pred_traj_fake_rel = model(model_input, obs_traj, seq_start_end, 0)
                l2_loss_rel.append(
                    l2_loss(
                        pred_traj_fake_rel,
                        model_input[-args.pred_len :],
                        loss_mask,
                        mode="raw",
                    )
                )

        l2_loss_sum_rel = torch.zeros(1).to(pred_traj_gt)
        l2_loss_rel = torch.stack(l2_loss_rel, dim=1)
        for start, end in seq_start_end.data:
            _l2_loss_rel = torch.narrow(l2_loss_rel, 0, start, end - start)
            _l2_loss_rel = torch.sum(_l2_loss_rel, dim=0)  # [20]
            _l2_loss_rel = torch.min(_l2_loss_rel) / (
                (pred_traj_fake_rel.shape[0]) * (end - start)
            )
            l2_loss_sum_rel += _l2_loss_rel

        loss += l2_loss_sum_rel
        losses.update(loss.item(), obs_traj.shape[1])
        loss.backward()
        optimizer.step()
        if batch_idx % args.print_every == 0:
            progress.display(batch_idx)
    writer.add_scalar("train_loss", losses.avg, epoch)


def validate(args, generator, val_loader, epoch, writer):
    ade = utils.AverageMeter("ADE", ":.6f")
    fde = utils.AverageMeter("FDE", ":.6f")
    progress = utils.ProgressMeter(len(val_loader), [ade, fde], prefix="Test: ")

    generator.eval()
    with torch.no_grad():
        for i, batch in enumerate(val_loader):
            batch = [tensor.cuda() for tensor in batch]
            (
                obs_traj,
                pred_traj_gt,
                obs_traj_rel,
                pred_traj_gt_rel,
                non_linear_ped,
                loss_mask,
                seq_start_end,
            ) = batch

            loss_mask = loss_mask[:, args.obs_len :]
            pred_traj_fake_rel = generator(obs_traj_rel, obs_traj, seq_start_end)

            pred_traj_fake_rel_predpart = pred_traj_fake_rel[-args.pred_len :]
            pred_traj_fake = relative_to_abs(pred_traj_fake_rel_predpart, obs_traj[-1])
            ade_, fde_ = cal_ade_fde(pred_traj_gt, pred_traj_fake)
            ade_ = ade_ / (obs_traj.shape[1] * args.pred_len)

            fde_ = fde_ / (obs_traj.shape[1])
            ade.update(ade_, obs_traj.shape[1])
            fde.update(fde_, obs_traj.shape[1])

            if i % args.print_every == 0:
                progress.display(i)

        logging.info(
            " * ADE  {ade.avg:.3f} FDE  {fde.avg:.3f}".format(ade=ade, fde=fde)
        )
        writer.add_scalar("val_ade", ade.avg, epoch)
    return ade.avg


def cal_ade_fde(pred_traj_gt, pred_traj_fake):
    ade = displacement_error(pred_traj_fake, pred_traj_gt)
    fde = final_displacement_error(pred_traj_fake[-1], pred_traj_gt[-1])
    return ade, fde


def save_checkpoint(state, save_ckpt, is_best, filename="checkpoint.pth.tar"):
    if is_best:
        torch.save(state, filename)
        logging.info("-------------- lower ade ----------------")
        ckpt_dir = os.path.split(filename)[0]
        shutil.copyfile(filename, ckpt_dir + "_model_best.pth.tar")
    elif save_ckpt:
        torch.save(state, filename)


if __name__ == "__main__":
    args = parser.parse_args()
    utils.set_logger(os.path.join(args.log_dir, f"train_{args.dataset_name}.log"))
    checkpoint_dir = f"./checkpoint_{args.dataset_name}"
    if os.path.exists(checkpoint_dir) is False:
        os.mkdir(checkpoint_dir)
    main(args)