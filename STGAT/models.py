import torch
import torch.nn as nn
import torch.nn.functional as F
import random


def get_noise(shape, noise_type):
    if noise_type == "gaussian":
        return torch.randn(*shape).cuda()
    elif noise_type == "uniform":
        return torch.rand(*shape).sub_(0.5).mul_(2.0).cuda()
    raise ValueError('Unrecognized noise type "%s"' % noise_type)


def make_mlp(dim_list, activation="leakyrelu", batch_norm=True, dropout=0, alpha=0.2):
    layers = []
    for dim_in, dim_out in zip(dim_list[:-1], dim_list[1:]):
        layers.append(nn.Linear(dim_in, dim_out))
        if batch_norm:
            layers.append(nn.BatchNorm1d(dim_out))
        if activation == "relu":
            layers.append(nn.ReLU())
        elif activation == "leakyrelu":
            layers.append(nn.LeakyReLU(negative_slope=alpha))
        if dropout > 0:
            layers.append(nn.Dropout(p=dropout))
    return nn.Sequential(*layers)


# this efficient implementation comes from https://github.com/xptree/DeepInf/
class BatchMultiHeadGraphAttention(nn.Module):
    def __init__(self, n_head, f_in, f_out, attn_dropout, bias=True):
        super(BatchMultiHeadGraphAttention, self).__init__()
        self.n_head = n_head
        self.f_in = f_in
        self.f_out = f_out
        self.w = nn.Parameter(torch.Tensor(n_head, f_in, f_out))
        self.a_src = nn.Parameter(torch.Tensor(n_head, f_out, 1))
        self.a_dst = nn.Parameter(torch.Tensor(n_head, f_out, 1))

        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2)
        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(attn_dropout)
        if bias:
            self.bias = nn.Parameter(torch.Tensor(f_out))
            nn.init.constant_(self.bias, 0)
        else:
            self.register_parameter("bias", None)

        nn.init.xavier_uniform_(self.w, gain=1.414)
        nn.init.xavier_uniform_(self.a_src, gain=1.414)
        nn.init.xavier_uniform_(self.a_dst, gain=1.414)

    def forward(self, h):
        bs, n = h.size()[:2]
        h_prime = torch.matmul(h.unsqueeze(1), self.w)
        attn_src = torch.matmul(h_prime, self.a_src)
        attn_dst = torch.matmul(h_prime, self.a_dst)
        attn = attn_src.expand(-1, -1, -1, n) + attn_dst.expand(-1, -1, -1, n).permute(
            0, 1, 3, 2
        )
        attn = self.leaky_relu(attn)
        attn = self.softmax(attn)
        attn = self.dropout(attn)
        output = torch.matmul(attn, h_prime)
        if self.bias is not None:
            return output + self.bias, attn
        else:
            return output, attn

    def __repr__(self):
        return (
            self.__class__.__name__
            + " ("
            + str(self.n_head)
            + " -> "
            + str(self.f_in)
            + " -> "
            + str(self.f_out)
            + ")"
        )


class GAT(nn.Module):
    def __init__(self, n_units, n_heads, dropout=0.2, alpha=0.2):
        super(GAT, self).__init__()
        self.n_layer = len(n_units) - 1
        self.dropout = dropout
        self.layer_stack = nn.ModuleList()

        for i in range(self.n_layer):
            f_in = n_units[i] * n_heads[i - 1] if i else n_units[i]
            self.layer_stack.append(
                BatchMultiHeadGraphAttention(
                    n_heads[i], f_in=f_in, f_out=n_units[i + 1], attn_dropout=dropout
                )
            )

        self.norm_list = [
            torch.nn.InstanceNorm1d(32).cuda(),
            torch.nn.InstanceNorm1d(64).cuda(),
        ]

    def forward(self, x):
        bs, n = x.size()[:2]
        for i, gat_layer in enumerate(self.layer_stack):
            x = self.norm_list[i](x.permute(0, 2, 1)).permute(0, 2, 1)
            x, attn = gat_layer(x)
            if i + 1 == self.n_layer:
                x = x.squeeze(dim=1)
            else:
                x = F.elu(x.transpose(1, 2).contiguous().view(bs, n, -1))
                x = F.dropout(x, self.dropout, training=self.training)
        else:
            return x


class GATEncoder(nn.Module):
    def __init__(self, n_units, n_heads, dropout, alpha):
        super(GATEncoder, self).__init__()
        self.gat_net = GAT(n_units, n_heads, dropout, alpha)

    def forward(self, obs_traj_embedding, seq_start_end):
        graph_embeded_data = []
        for start, end in seq_start_end.data:
            curr_seq_embedding_traj = obs_traj_embedding[:, start:end, :]
            curr_seq_graph_embedding = self.gat_net(curr_seq_embedding_traj)
            graph_embeded_data.append(curr_seq_graph_embedding)
        graph_embeded_data = torch.cat(graph_embeded_data, dim=1)
        return graph_embeded_data


class Encoder(nn.Module):
    """Encoder is part of both TrajectoryGenerator and
    TrajectoryDiscriminator"""

    def __init__(
        self,
        obs_len,
        pred_len,
        traj_lstm_input_size,
        traj_lstm_hidden_size,
        n_units,
        n_heads,
        graph_network_out_dims,
        graph_lstm_hidden_size,
        noise_dim=(8,),
        noise_type="gaussian",
        dropout=0.2,
        alpha=0.2
    ):
        super(Encoder, self).__init__()

        self.obs_len = obs_len
        self.pred_len = pred_len

        self.gatencoder = GATEncoder(
            n_units=n_units, n_heads=n_heads, dropout=dropout, alpha=alpha
        )

        self.graph_lstm_hidden_size = graph_lstm_hidden_size
        self.traj_lstm_hidden_size = traj_lstm_hidden_size
        self.traj_lstm_input_size = traj_lstm_input_size

        self.noise_dim = noise_dim
        self.noise_type = noise_type

    def init_hidden_traj_lstm(self, batch):
        return (
            torch.randn(batch, self.traj_lstm_hidden_size).cuda(),
            torch.randn(batch, self.traj_lstm_hidden_size).cuda(),
        )

    def init_hidden_graph_lstm(self, batch):
        return (
            torch.randn(batch, self.graph_lstm_hidden_size).cuda(),
            torch.randn(batch, self.graph_lstm_hidden_size).cuda(),
        )

    def add_noise(self, _input, seq_start_end):
        noise_shape = (seq_start_end.size(0),) + self.noise_dim

        z_decoder = get_noise(noise_shape, self.noise_type)

        _list = []
        for idx, (start, end) in enumerate(seq_start_end):
            start = start.item()
            end = end.item()
            _vec = z_decoder[idx].view(1, -1)
            _to_cat = _vec.repeat(end - start, 1)
            _list.append(torch.cat([_input[start:end], _to_cat], dim=1))
        decoder_h = torch.cat(_list, dim=0)
        return decoder_h

    def forward(
        self,
        obs_traj_rel,
        obs_traj_pos,
        seq_start_end,
    ):
        batch = obs_traj_rel.shape[1]
        traj_lstm_h_t, traj_lstm_c_t = self.init_hidden_traj_lstm(batch)
        graph_lstm_h_t, graph_lstm_c_t = self.init_hidden_graph_lstm(batch)

        pred_traj_rel = []
        traj_lstm_hidden_states = []
        graph_lstm_hidden_states = []
        for i, input_t in enumerate(
            obs_traj_rel[: self.obs_len].chunk(
                obs_traj_rel[: self.obs_len].size(0), dim=0
            )
        ):
            traj_lstm_h_t, traj_lstm_c_t = self.traj_lstm_model(
                input_t.squeeze(0), (traj_lstm_h_t, traj_lstm_c_t)
            )
            traj_lstm_hidden_states += [traj_lstm_h_t]

        graph_lstm_input = self.gatencoder(
            torch.stack(traj_lstm_hidden_states), seq_start_end
        )
        for i, input_t in enumerate(
            graph_lstm_input[: self.obs_len].chunk(
                graph_lstm_input[: self.obs_len].size(0), dim=0
            )
        ):
            graph_lstm_h_t, graph_lstm_c_t = self.graph_lstm_model(
                input_t.squeeze(0), (graph_lstm_h_t, graph_lstm_c_t)
            )
            graph_lstm_hidden_states += [graph_lstm_h_t]

        encoded_before_noise_hidden = torch.cat(
            (traj_lstm_hidden_states[-1], graph_lstm_hidden_states[-1]), dim=1
        )

        decoder_hidden = self.add_noise(encoded_before_noise_hidden, seq_start_end)
        return decoder_hidden


class Decoder(nn.Module):
    """Decoder is part of both TrajectoryGenerator and"""

    def __init__(
        self,
        obs_len,
        pred_len,
        traj_lstm_input_size,
        traj_lstm_hidden_size,
        graph_lstm_hidden_size,
        noise_dim=(8,),
    ):
        super(Decoder, self).__init__()

        pred_lstm_hidden_size = (
            traj_lstm_hidden_size + graph_lstm_hidden_size + noise_dim[0]
        )
        self.pred_lstm_model = nn.LSTMCell(
            traj_lstm_input_size, pred_lstm_hidden_size
        )
        self.pred_hidden2pos = nn.Linear(pred_lstm_hidden_size, 2)

    def forward(
        self,
        pred_lstm_hidden,
        obs_traj_rel,
        obs_traj_pos,
        teacher_forcing_ratio=0.5,
    ):
        pred_lstm_c_t = torch.zeros_like(pred_lstm_hidden).cuda()
        output = obs_traj_rel[self.obs_len - 1]

        if self.training:
            for i, input_t in enumerate(
                obs_traj_rel[-self.pred_len :].chunk(
                    obs_traj_rel[-self.pred_len :].size(0), dim=0
                )
            ):
                teacher_force = random.random() < teacher_forcing_ratio
                input_t = input_t if teacher_force else output.unsqueeze(0)
                pred_lstm_hidden, pred_lstm_c_t = self.pred_lstm_model(
                    input_t.squeeze(0), (pred_lstm_hidden, pred_lstm_c_t)
                )
                output = self.pred_hidden2pos(pred_lstm_hidden)
                pred_traj_rel += [output]

            outputs = torch.stack(pred_traj_rel)
        else:
            for i in range(self.pred_len):
                pred_lstm_hidden, pred_lstm_c_t = self.pred_lstm_model(
                    output, (pred_lstm_hidden, pred_lstm_c_t)
                )
                output = self.pred_hidden2pos(pred_lstm_hidden)
                pred_traj_rel += [output]

            outputs = torch.stack(pred_traj_rel)

        return outputs


class TrajectoryGenerator(nn.Module):
    def __init__(
        self,
        obs_len,
        pred_len,
        traj_lstm_input_size,
        traj_lstm_hidden_size,
        n_units,
        n_heads,
        graph_network_out_dims,
        dropout,
        alpha,
        graph_lstm_hidden_size,
        noise_dim=(8,),
        noise_type="gaussian",
    ):
        super(TrajectoryGenerator, self).__init__()

        self.encoder = Encoder(
            obs_len=obs_len,
            pred_len=pred_len,
            traj_lstm_input_size=traj_lstm_input_size,
            traj_lstm_hidden_size=traj_lstm_hidden_size,
            n_units=n_units,
            n_heads=n_heads,
            graph_network_out_dims=graph_network_out_dims,
            graph_lstm_hidden_size=graph_lstm_hidden_size,
            noise_dim=noise_dim,
            noise_type=noise_type,
            dropout=dropout,
        )

        self.decoder = Decoder(
            obs_len=obs_len,
            pred_len=pred_len,
            traj_lstm_input_size=traj_lstm_input_size,
            traj_lstm_hidden_size=traj_lstm_hidden_size,
            graph_lstm_hidden_size=graph_lstm_hidden_size,
            noise_dim=noise_dim,
        )

    def forward(
        self,
        obs_traj_rel,
        obs_traj_pos,
        seq_start_end,
        teacher_forcing_ratio=0.5,
        # training_step=3,
    ):
        pred_lstm_hidden = self.encoder(obs_traj_rel, obs_traj_pos, seq_start_end)
        outputs = self.decoder(
            pred_lstm_hidden, obs_traj_rel, obs_traj_pos, teacher_forcing_ratio=0.5
        )

        return outputs


class TrajectoryDiscriminator(nn.Module):
    def __init__(
        self,
        obs_len,
        pred_len,
        traj_lstm_input_size,
        traj_lstm_hidden_size,
        n_units,
        n_heads,
        graph_network_out_dims,
        graph_lstm_hidden_size,
        mlp_dim,
        noise_dim=(8,),
        noise_type="gaussian",
        activation="leakyrelu",
        batch_norm=True,
        dropout=0.2,
        alpha=0.2,
    ):
        super(TrajectoryDiscriminator, self).__init__()

        self.encoder = Encoder(
            obs_len=obs_len,
            pred_len=pred_len,
            traj_lstm_input_size=traj_lstm_input_size,
            traj_lstm_hidden_size=traj_lstm_hidden_size,
            n_units=n_units,
            n_heads=n_heads,
            graph_network_out_dims=graph_network_out_dims,
            graph_lstm_hidden_size=graph_lstm_hidden_size,
            noise_dim=noise_dim,
            noise_type=noise_type,
            dropout=dropout,
        )

        pred_lstm_hidden_size = (
            traj_lstm_hidden_size + graph_lstm_hidden_size + noise_dim[0]
        )
        real_classifier_dims = [pred_lstm_hidden_size, mlp_dim, 1]
        self.real_classifier = make_mlp(
            real_classifier_dims,
            activation=activation,
            batch_norm=batch_norm,
            dropout=dropout,
        )

    def forward(
        self,
        obs_traj_rel,
        obs_traj_pos,
        seq_start_end,
        teacher_forcing_ratio=0.5,
        # training_step=3,
    ):
        classifier_input = self.encoder(obs_traj_rel, obs_traj_pos, seq_start_end)
        scores = self.real_classifier(classifier_input)

        return scores
