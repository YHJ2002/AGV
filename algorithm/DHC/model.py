import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence
from torch.cuda.amp import autocast
from . import configs


class ResBlock(nn.Module):
    def __init__(self, channel):
        super().__init__()

        self.block1 = nn.Conv2d(channel, channel, 3, 1, 1)
        self.block2 = nn.Conv2d(channel, channel, 3, 1, 1)

    def forward(self, x):
        identity = x

        x = self.block1(x)
        x = F.relu(x)

        x = self.block2(x)

        # 残差连接，帮助卷积编码器更稳定地训练。
        x += identity

        x = F.relu(x)

        return x


class MultiHeadAttention(nn.Module):
    def __init__(self, input_dim, output_dim, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.W_Q = nn.Linear(input_dim, output_dim * num_heads)
        self.W_K = nn.Linear(input_dim, output_dim * num_heads)
        self.W_V = nn.Linear(input_dim, output_dim * num_heads)
        self.W_O = nn.Linear(output_dim * num_heads, output_dim, bias=False)

    def forward(self, input, attn_mask):
        # input: [batch_size, num_agents, input_dim]
        batch_size, num_agents, input_dim = input.size()
        assert input_dim == self.input_dim

        # 线性投影后拆成多头，再转成注意力计算需要的维度顺序。
        q_s = self.W_Q(input).view(batch_size, num_agents, self.num_heads, -1).transpose(1, 2)
        k_s = self.W_K(input).view(batch_size, num_agents, self.num_heads, -1).transpose(1, 2)
        v_s = self.W_V(input).view(batch_size, num_agents, self.num_heads, -1).transpose(1, 2)

        if attn_mask.dim() == 2:
            attn_mask = attn_mask.unsqueeze(0)
        assert attn_mask.size(0) == batch_size, 'mask dim {} while batch size {}'.format(attn_mask.size(0), batch_size)

        # mask 为 True 的位置会被屏蔽，不能参与注意力计算。
        attn_mask = attn_mask.unsqueeze(1).repeat_interleave(self.num_heads, 1)
        assert attn_mask.size() == (batch_size, self.num_heads, num_agents, num_agents)

        # 在 fp32 下计算分数，减少混合精度下的数值不稳定。
        with autocast(enabled=False):
            scores = torch.matmul(q_s.float(), k_s.float().transpose(-1, -2)) / (self.output_dim**0.5)
            scores.masked_fill_(attn_mask, -1e9)
            attn = F.softmax(scores, dim=-1)

        context = torch.matmul(attn, v_s)
        context = context.transpose(1, 2).contiguous().view(batch_size, num_agents, self.num_heads * self.output_dim)
        output = self.W_O(context)

        return output


class CommBlock(nn.Module):
    def __init__(self, input_dim, output_dim=64, num_heads=configs.num_comm_heads, num_layers=configs.num_comm_layers):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.self_attn = MultiHeadAttention(input_dim, output_dim, num_heads)

        self.update_cell = nn.GRUCell(output_dim, input_dim)

    def forward(self, latent, comm_mask):
        '''
        latent: [batch_size, num_agents, latent_dim]
        comm_mask: [batch_size, num_agents, num_agents]
        '''
        num_agents = latent.size(1)

        # 只有通信邻居数大于 1 的 agent 才需要更新隐藏状态。
        update_mask = comm_mask.sum(dim=-1) > 1
        comm_idx = update_mask.nonzero(as_tuple=True)

        # 没有 agent 需要通信时直接返回。
        if len(comm_idx[0]) == 0:
            return latent

        if len(comm_idx) > 1:
            update_mask = update_mask.unsqueeze(2)

        attn_mask = comm_mask == False

        for _ in range(self.num_layers):
            # 每层先聚合邻居信息，再通过 GRUCell 融合到旧状态。
            info = self.self_attn(latent, attn_mask=attn_mask)
            if len(comm_idx) == 1:
                batch_idx = torch.zeros(len(comm_idx[0]), dtype=torch.long)
                latent[batch_idx, comm_idx[0]] = self.update_cell(info[batch_idx, comm_idx[0]], latent[batch_idx, comm_idx[0]])
            else:
                update_info = self.update_cell(
                    info.view(-1, self.output_dim),
                    latent.view(-1, self.input_dim)
                ).view(configs.batch_size, num_agents, self.input_dim)
                latent = torch.where(update_mask, update_info, latent)

        return latent


class Network(nn.Module):
    def __init__(self, input_shape=configs.obs_shape, cnn_channel=configs.cnn_channel, hidden_dim=configs.hidden_dim,
                max_comm_agents=configs.max_comm_agents):

        super().__init__()

        self.input_shape = input_shape
        self.latent_dim = 16 * 7 * 7
        self.hidden_dim = hidden_dim
        self.max_comm_agents = max_comm_agents

        # 局部观测先经过 CNN 编码成固定长度向量。
        self.obs_encoder = nn.Sequential(
            nn.Conv2d(self.input_shape[0], cnn_channel, 3, 1),
            nn.ReLU(True),

            ResBlock(cnn_channel),

            ResBlock(cnn_channel),

            ResBlock(cnn_channel),

            nn.Conv2d(cnn_channel, 16, 1, 1),
            nn.ReLU(True),

            nn.Flatten(),
        )

        self.recurrent = nn.GRUCell(self.latent_dim, self.hidden_dim)

        # 通信模块负责在 agent 之间交换隐藏状态信息。
        self.comm = CommBlock(hidden_dim)

        # Dueling DQN 结构
        self.adv = nn.Linear(hidden_dim, 5)
        self.state = nn.Linear(hidden_dim, 1)

        self.hidden = None

        for _, m in self.named_modules():
            if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    @torch.no_grad()
    def step(self, obs, pos):
        num_agents = obs.size(0)

        latent = self.obs_encoder(obs)

        # 推理阶段沿用上一步隐藏状态。
        if self.hidden is None:
            self.hidden = self.recurrent(latent)
        else:
            self.hidden = self.recurrent(latent, self.hidden)

        # 补一维 batch 维度，便于复用通信模块。
        self.hidden = self.hidden.unsqueeze(0)

        # 根据空间位置构造通信 mask。
        agents_pos = pos
        pos_mat = (agents_pos.unsqueeze(1) - agents_pos.unsqueeze(0)).abs()
        dist_mat = (pos_mat[:, :, 0]**2 + pos_mat[:, :, 1]**2).sqrt()

        # 超出局部视野范围的 agent 不可通信。
        in_obs_mask = (pos_mat <= configs.obs_radius).all(2)
        # 在视野内也只保留最近的若干个通信对象。
        _, ranking = dist_mat.topk(min(self.max_comm_agents, num_agents), dim=1, largest=False)
        dist_mask = torch.zeros((num_agents, num_agents), dtype=torch.bool, device=obs.device)
        dist_mask.scatter_(1, ranking, True)

        comm_mask = torch.bitwise_and(in_obs_mask, dist_mask)

        self.hidden = self.comm(self.hidden, comm_mask)
        self.hidden = self.hidden.squeeze(0)

        adv_val = self.adv(self.hidden)
        state_val = self.state(self.hidden)

        q_val = state_val + adv_val - adv_val.mean(1, keepdim=True)

        actions = torch.argmax(q_val, 1).tolist()

        return (
            actions,
            q_val.cpu().numpy(),
            self.hidden.cpu().numpy(),
            comm_mask.cpu().numpy()
        )

    def reset(self):
        self.hidden = None

    @autocast()
    def forward(self, obs, steps, hidden, comm_mask):
        # obs: [batch_size, seq_len(+n_step), num_agents, C, H, W]
        # comm_mask: [batch_size, seq_len(+n_step), max_num_agents, max_num_agents]
        max_steps = obs.size(1)
        num_agents = comm_mask.size(2)

        assert comm_mask.size(2) == configs.max_num_agents

        obs = obs.transpose(1, 2)
        obs = obs.contiguous().view(-1, *self.input_shape)

        latent = self.obs_encoder(obs)
        latent = latent.view(configs.batch_size * num_agents, max_steps, self.latent_dim).transpose(0, 1)

        hidden_buffer = []
        for i in range(max_steps):
            # 先做时间维上的 GRU 更新，再做 agent 间通信。
            hidden = self.recurrent(latent[i], hidden)
            hidden = hidden.view(configs.batch_size, num_agents, self.hidden_dim)
            hidden = self.comm(hidden, comm_mask[:, i])

            # 当前训练逻辑只使用 agent 0 的隐藏状态来估计 Q。
            hidden_buffer.append(hidden[:, 0])
            hidden = hidden.view(configs.batch_size * num_agents, self.hidden_dim)

        hidden_buffer = torch.stack(hidden_buffer).transpose(0, 1)

        # 根据每条样本的实际长度取最后一个有效时间步。
        hidden = hidden_buffer[torch.arange(configs.batch_size), steps - 1]

        adv_val = self.adv(hidden)
        state_val = self.state(hidden)

        q_val = state_val + adv_val - adv_val.mean(1, keepdim=True)

        return q_val
