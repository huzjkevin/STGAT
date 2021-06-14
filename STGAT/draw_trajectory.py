import argparse
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import random
import torch
import matplotlib.pyplot as plt
import numpy as np

plt.style.use("seaborn-dark")
import scipy
import torch

from transformer.batch import subsequent_mask
import utils
from data.loader import data_loader
from models import TrajectoryGenerator
from utils import (
    displacement_error,
    final_displacement_error,
    get_dset_path,
    int_tuple,
    l2_loss,
    relative_to_abs,
)
import individual_TF


parser = argparse.ArgumentParser()
parser.add_argument("--log_dir", default="./", help="Directory containing logging file")

parser.add_argument("--dataset_name", default="trajectory_combined", type=str)
parser.add_argument("--delim", default="\t")
parser.add_argument("--loader_num_workers", default=16, type=int)
parser.add_argument("--obs_len", default=8, type=int)
parser.add_argument("--pred_len", default=8, type=int)
parser.add_argument("--skip", default=1, type=int)

parser.add_argument("--seed", type=int, default=72, help="Random seed.")
parser.add_argument("--batch_size", default=32, type=int)

parser.add_argument("--noise_dim", default=(16,), type=int_tuple)
parser.add_argument("--noise_type", default="gaussian")
parser.add_argument("--noise_mix_type", default="global")

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


parser.add_argument("--num_samples", default=20, type=int)


parser.add_argument(
    "--dropout", type=float, default=0, help="Dropout rate (1 - keep probability)."
)
parser.add_argument(
    "--alpha", type=float, default=0.2, help="Alpha for the leaky_relu."
)

parser.add_argument("--dset_type", default="test", type=str)


parser.add_argument(
    "--resume",
    default="checkpoint_trajectory_combined_model_best.pth.tar",
    type=str,
    metavar="PATH",
    help="path to latest checkpoint (default: none)",
)

warmup = 10
emb_size = 512

def evaluate_helper(error, seq_start_end, model_output_traj, model_output_traj_best):
    error = torch.stack(error, dim=1)
    for (start, end) in seq_start_end:
        start = start.item()
        end = end.item()
        _error = error[start:end]
        _error = torch.sum(_error, dim=0)
        min_index = _error.min(0)[1].item()
        model_output_traj_best[:, start:end, :] = model_output_traj[min_index][
            :, start:end, :
        ]
    return model_output_traj_best


def get_generator(checkpoint):
    n_units = (
        [args.traj_lstm_hidden_size]
        + [int(x) for x in args.hidden_units.strip().split(",")]
        + [args.graph_lstm_hidden_size]
    )
    n_heads = [int(x) for x in args.heads.strip().split(",")]
    model = individual_TF.IndividualTF(
        2,
        3,
        3,
        N=6,
        d_model=emb_size,
        d_ff=2048,
        h=8,
        dropout=0.1,
        mean=[0, 0],
        std=[0, 0],
    )
    model.load_state_dict(checkpoint["state_dict"])
    model.cuda()
    model.eval()
    return model


def cal_ade_fde(pred_traj_gt, pred_traj_fake):
    ade = displacement_error(pred_traj_fake, pred_traj_gt, mode="raw")
    fde = final_displacement_error(pred_traj_fake[-1], pred_traj_gt[-1], mode="raw")
    return ade, fde


def plot_trajectory(args, loader, generator):
    mat = scipy.io.loadmat("norm.mat")
    mean = torch.Tensor(mat["mean"]).cuda()
    std = torch.Tensor(mat["std"]).cuda()
    ground_truth_input = []
    all_model_output_traj = []
    ground_truth_output = []
    pic_cnt = 0
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
            ) = batch
            ade = []
            ground_truth_input.append(obs_traj)
            ground_truth_output.append(pred_traj_gt)
            model_output_traj = []
            model_output_traj_best = torch.ones_like(pred_traj_gt).cuda()

            for _ in range(args.num_samples):
                # for transformers
                inp = (obs_traj_rel.permute(1, 0, 2)[:, 1:, :] - mean) / std
                src_att = torch.ones((inp.shape[0], 1, inp.shape[1])).cuda()
                start_of_seq = (
                    torch.Tensor([0, 0, 1])
                    .unsqueeze(0)
                    .unsqueeze(1)
                    .repeat(inp.shape[0], 1, 1)
                    .cuda()
                )
                dec_inp = start_of_seq

                for i in range(args.pred_len):
                    trg_att = (
                        subsequent_mask(dec_inp.shape[1])
                        .repeat(dec_inp.shape[0], 1, 1)
                        .cuda()
                    )
                    out = generator(inp, dec_inp, src_att, trg_att)
                    dec_inp = torch.cat((dec_inp, out[:, -1:, :]), 1)

                preds_tr_b = dec_inp[:, 1:, 0:2] * std + mean
                pred_traj_fake_rel = preds_tr_b.permute(1, 0, 2)
                pred_traj_fake_rel = pred_traj_fake_rel[:, :, 0:2]
                pred_traj_fake_rel = pred_traj_fake_rel[-args.pred_len :]

                pred_traj_fake = relative_to_abs(pred_traj_fake_rel, obs_traj[-1])
                model_output_traj.append(pred_traj_fake)
                ade_, fde_ = cal_ade_fde(pred_traj_gt, pred_traj_fake)
                ade.append(ade_)

            model_output_traj_best = evaluate_helper(
                ade, seq_start_end, model_output_traj, model_output_traj_best
            )
            all_model_output_traj.append(model_output_traj_best)

            for (start, end) in seq_start_end:
                plt.figure(figsize=(20,15), dpi=100)
                ground_truth_input_x_piccoor = (
                    obs_traj[:, start:end, :].cpu().numpy()[:, :, 0].T
                )
                ground_truth_input_y_piccoor = (
                    obs_traj[:, start:end, :].cpu().numpy()[:, :, 1].T
                )
                ground_truth_output_x_piccoor = (
                    pred_traj_gt[:, start:end, :].cpu().numpy()[:, :, 0].T
                )
                ground_truth_output_y_piccoor = (
                    pred_traj_gt[:, start:end, :].cpu().numpy()[:, :, 1].T
                )
                model_output_x_piccoor = (
                    model_output_traj_best[:, start:end, :].cpu().numpy()[:, :, 0].T
                )
                model_output_y_piccoor = (
                    model_output_traj_best[:, start:end, :].cpu().numpy()[:, :, 1].T
                )
                for i in range(ground_truth_output_x_piccoor.shape[0]):

                    observed_line = plt.plot(
                        ground_truth_input_x_piccoor[i, :],
                        ground_truth_input_y_piccoor[i, :],
                        "r-",
                        linewidth=4,
                        label="Observed Trajectory",
                    )[0]
                    observed_line.axes.annotate(
                        "",
                        xytext=(
                            ground_truth_input_x_piccoor[i, -2],
                            ground_truth_input_y_piccoor[i, -2],
                        ),
                        xy=(
                            ground_truth_input_x_piccoor[i, -1],
                            ground_truth_input_y_piccoor[i, -1],
                        ),
                        arrowprops=dict(
                            arrowstyle="->", color=observed_line.get_color(), lw=1
                        ),
                        size=20,
                    )
                    ground_line = plt.plot(
                        np.append(
                            ground_truth_input_x_piccoor[i, -1],
                            ground_truth_output_x_piccoor[i, :],
                        ),
                        np.append(
                            ground_truth_input_y_piccoor[i, -1],
                            ground_truth_output_y_piccoor[i, :],
                        ),
                        "b-",
                        linewidth=4,
                        label="Ground Truth",
                    )[0]
                    predict_line = plt.plot(
                        np.append(
                            ground_truth_input_x_piccoor[i, -1],
                            model_output_x_piccoor[i, :],
                        ),
                        np.append(
                            ground_truth_input_y_piccoor[i, -1],
                            model_output_y_piccoor[i, :],
                        ),
                        color="#ffff00",
                        ls="--",
                        linewidth=4,
                        label="Predicted Trajectory",
                    )[0]

                #plt.axis("off")
                plt.savefig(
                    "./traj_fig_{}/pic_{}.png".format(args.dataset_name, pic_cnt)
                )
                plt.close()
                pic_cnt += 1


def main(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    checkpoint = torch.load(args.resume)
    generator = get_generator(checkpoint)
    path = get_dset_path(args.dataset_name, args.dset_type)

    _, loader = data_loader(args, path)
    plot_trajectory(args, loader, generator)


if __name__ == "__main__":
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.mkdir(f"./traj_fig_{args.dataset_name}")
    main(args)
