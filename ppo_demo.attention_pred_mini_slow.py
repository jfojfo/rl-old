import os
from collections import OrderedDict
import gymnasium as gym
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from torch.distributions import Categorical
from torch.utils.tensorboard import SummaryWriter
from stable_baselines3.common.vec_env import VecVideoRecorder, SubprocVecEnv
import argparse
from torchinfo import summary
import torch.nn.functional as F
import torchvision
import time


#
# https://github.com/rossettisimone/PPO_PONG_DISCRETE/blob/master/PPO_PONG.ipynb
#


H_SIZE = 32 # hidden size, linear units of the output layer
L_RATE = 1e-4 # learning rate, gradient coefficient for CNN weight update
G_GAE = 0.99 # gamma param for GAE
L_GAE = 0.95 # lambda param for GAE
E_CLIP = 0.2 # clipping coefficient
C_1 = 0.5 # squared loss coefficient
C_2 = 0.01 # entropy coefficient
C_3 = 1 # predict loss coefficient
N = 8 # simultaneous processing environments
T = 256 # PPO steps to get envs data
M = 64 # mini batch size
K = 10 # PPO epochs repeated to optimise
T_EPOCHS = 50 # T_EPOCH to test and save
N_TESTS = 10 # do N_TESTS tests
TARGET_REWARD = 20
TRANSFER_LEARNING = False
EMBED_DIM = H_SIZE
LOOK_BACK_SIZE = 16

MODEL_DIR = 'models'
MODEL = f'ppo_demo.attention_pred_mini_attn_grad_M_{M}_lookback_{LOOK_BACK_SIZE}_c3_{C_3}_bce'
# ENV_ID = 'Pong-v0'
ENV_ID = 'PongDeterministic-v0'


class MySelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=None):
        super(MySelfAttention, self).__init__()

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout

        # Linear transformations for Query, Key, and Value
        self.query_transform = nn.Linear(embed_dim, embed_dim)
        self.key_transform = nn.Linear(embed_dim, embed_dim)
        self.value_transform = nn.Linear(embed_dim, embed_dim)

        # Output linear transformation
        self.output_transform = nn.Linear(embed_dim, embed_dim)

    def forward(self, q_inputs, k_inputs, v_inputs, key_padding_mask=None):
        # Apply linear transformations to inputs to get Query, Key, and Value
        query = self.query_transform(q_inputs)
        key = self.key_transform(k_inputs)
        value = self.value_transform(v_inputs)

        # Reshape Query, Key, and Value tensors
        batch_size, seq_len, embed_dim = q_inputs.size()
        query = query.view(batch_size, seq_len, self.num_heads, -1).transpose(1, 2)

        batch_size, seq_len, embed_dim = k_inputs.size()
        key = key.view(batch_size, seq_len, self.num_heads, -1).transpose(1, 2)

        batch_size, seq_len, embed_dim = v_inputs.size()
        value = value.view(batch_size, seq_len, self.num_heads, -1).transpose(1, 2)

        # Compute scaled dot-product attention
        scores = torch.matmul(query, key.transpose(-2, -1)) / (self.embed_dim ** 0.5)

        # Apply key padding mask if provided
        if key_padding_mask is not None:
            scores = scores.masked_fill(key_padding_mask.unsqueeze(1).unsqueeze(2), float('-inf'))

        attention_weights = torch.softmax(scores, dim=-1)
        if self.dropout is not None:
            attention_weights = F.dropout(attention_weights, p=self.dropout, training=self.training)

        attention_output = torch.matmul(attention_weights, value)

        # Reshape and transpose attention output
        attention_output = attention_output.transpose(1, 2).contiguous().view(batch_size, seq_len, embed_dim)

        # Apply output linear transformation
        output = self.output_transform(attention_output)
        return output


class MyTransformerEncoderLayer(nn.Module):
    def __init__(self, embed_dim, num_heads, dim_feedforward=1024, dropout=0.1):
        super(MyTransformerEncoderLayer, self).__init__()
        """
        :param d_model:         d_k = d_v = d_model/nhead = 64, 模型中向量的维度，论文默认值为 512
        :param nhead:           多头注意力机制中多头的数量，论文默认为值 8
        :param dim_feedforward: 全连接中向量的维度，论文默认值为 2048
        :param dropout:         丢弃率，论文中的默认值为 0.1    
        """
        # self.self_attn = MySelfAttention(embed_dim, num_heads, dropout=dropout)
        self.self_attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(embed_dim)

        self.linear1 = nn.Linear(embed_dim, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, embed_dim)
        self.activation = F.relu

        self.dropout2 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(embed_dim)

    def forward(self, query, key, value, src_mask=None, src_key_padding_mask=None):
        """
        :param src: 编码部分的输入，形状为 [src_len,batch_size, embed_dim]
        :param src_mask:  编码部分输入的padding情况，形状为 [batch_size, src_len]
        :return: # [src_len, batch_size, num_heads * kdim] <==> [src_len,batch_size,embed_dim]
        """
        src = query
        src2, w = self.self_attn(query, key, value, attn_mask=src_mask, key_padding_mask=src_key_padding_mask)  # 计算多头注意力
        # src2: [src_len,batch_size,num_heads*kdim] num_heads*kdim = embed_dim
        src = src + self.dropout1(src2)  # 残差连接
        src = self.norm1(src)  # [src_len,batch_size,num_heads*kdim]

        src2 = self.activation(self.linear1(src))  # [src_len,batch_size,dim_feedforward]
        src2 = self.linear2(self.dropout(src2))  # [src_len,batch_size,num_heads*kdim]
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src, w  # [src_len, batch_size, num_heads * kdim] <==> [src_len,batch_size,embed_dim]


class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_length=512):
        super(PositionalEmbedding, self).__init__()
        self.d_model = d_model

        # Create constant positional encoding matrix
        pe = torch.zeros(max_length, d_model)

        position = torch.arange(0, max_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))

        # Calculate the positional encodings
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        # Register the positional encoding as a buffer
        self.register_buffer('pe', pe.unsqueeze(1))

    def forward(self, x):
        # Add positional encodings to the input tensor
        x = x + self.pe[:x.size(0), :]
        return x


class MyPositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_length=128):
        super(MyPositionalEmbedding, self).__init__()
        self.d_model = d_model

        # Create constant positional encoding matrix
        pe = torch.zeros(max_length, d_model)

        position = torch.arange(0, max_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))

        # Calculate the positional encodings
        pe = torch.exp(-position)

        # Register the positional encoding as a buffer
        self.register_buffer('pe', pe.unsqueeze(1))

    def forward(self, x):
        # Add positional encodings to the input tensor
        x = x * self.pe[:x.size(0), :]
        return x


class ActorCritic(nn.Module):
    def __init__(self, num_inputs, num_outputs, hidden_size=H_SIZE, embed_dim=EMBED_DIM):
        super(ActorCritic, self).__init__()
        self.num_outputs = num_outputs
        self.feature = nn.Sequential(OrderedDict([
            ('conv1', nn.Conv2d(in_channels=num_inputs, out_channels=16, kernel_size=8, stride=4)),
            # ('batchnorm1', nn.BatchNorm2d(16)),
            ('act1', nn.ReLU()),
            ('conv2', nn.Conv2d(in_channels=16, out_channels=32, kernel_size=4, stride=2)),
            # ('batchnorm2', nn.BatchNorm2d(32)),
            ('act2', nn.ReLU()),
            ('conv3', nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=2)),
            ('act3', nn.ReLU()),
            ('flatten', nn.Flatten()),
            ('linear', nn.Linear(in_features=32 * 4 * 4, out_features=hidden_size)),
            # ('softmax', nn.Softmax(dim=-1)),
            # ('norm', nn.LayerNorm(hidden_size)),
            ('act', nn.Sigmoid()),  # 加sigmoid/softmax，防止越来越小
        ]))

        # self.decoder = nn.Sequential(OrderedDict([
        #     ('linear', nn.Linear(in_features=hidden_size + num_outputs, out_features=16 * 4 * 4)),
        #     ('unflatten', nn.Unflatten(1, (16, 4, 4))),
        #     ('act1', nn.ReLU()),
        #     ('unconv1', nn.ConvTranspose2d(16, 32, kernel_size=3, stride=2)),
        #     ('act2', nn.ReLU()),
        #     ('unconv2', nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2)),
        #     ('act3', nn.ReLU()),
        #     ('unconv3', nn.ConvTranspose2d(16, 1, kernel_size=8, stride=4)),  # logits
        # ]))

        # self.attention = nn.MultiheadAttention(embed_dim, 16)
        self.attention = MyTransformerEncoderLayer(embed_dim, 16, embed_dim * 2)
        self.pos_encode = MyPositionalEmbedding(EMBED_DIM, LOOK_BACK_SIZE + 1)

        self.pred_next = nn.Sequential(
            nn.Linear(in_features=hidden_size + num_outputs, out_features=hidden_size * 4),
            nn.ReLU(),
            nn.Linear(in_features=hidden_size * 4, out_features=hidden_size),  # logits
        )

        self.critic = nn.Sequential(  # The “Critic” estimates the value function
            # nn.Linear(in_features=hidden_size, out_features=hidden_size),
            # nn.ReLU(),
            nn.Linear(in_features=hidden_size, out_features=1),
        )
        self.actor = nn.Sequential(  # The “Actor” updates the policy distribution in the direction suggested by the Critic (such as with policy gradients)
            # nn.Linear(in_features=hidden_size, out_features=hidden_size),
            # nn.ReLU(),
            nn.Linear(in_features=hidden_size, out_features=num_outputs),
            nn.Softmax(dim=1),
        )

    def forward(self, x, memory, mask=None):
        feature = self.feature(x)
        query = feature.unsqueeze(dim=0) # sequence first
        memory_concat = torch.cat([query.detach(), memory], dim=0)
        # memory_concat_pos = self.pos_encode(memory_concat)
        if mask is not None:
            mask_concat = torch.cat([torch.zeros((mask.shape[0], 1), dtype=mask.dtype).to(mask.device), mask], dim=1)
        else:
            mask_concat = None
        # query_pos = self.pos_encode(query)
        # attention, scores = self.attention(query_pos, memory_pos, memory_pos, src_key_padding_mask=mask)
        attention, scores = self.attention(query, memory_concat, memory_concat, src_key_padding_mask=mask_concat)
        attn_out = attention.squeeze(dim=0)
        value = self.critic(attn_out)
        probs = self.actor(attn_out)
        dist = Categorical(probs)

        return dist, value, feature, attn_out

    def pred_next_feature(self, attn_out, action, x_next):
        feature_next = self.feature(x_next)
        action_one_hot = F.one_hot(action.squeeze(dim=1), self.num_outputs).to(torch.float32)
        mixed = torch.concat((attn_out, action_one_hot), dim=1)
        pred_feature_next_logits = self.pred_next(mixed)
        return pred_feature_next_logits, feature_next

    def pred_next_state(self, attn_out, action):
        action_one_hot = F.one_hot(action.squeeze(dim=1), self.num_outputs).to(torch.float32)
        mixed = torch.concat((attn_out, action_one_hot), dim=1)
        x_recon = self.decoder(mixed)
        return x_recon


# class SeqTorch:
#     def __init__(self, seq_len=LOOK_BACK_SIZE, batch_size=N, rolling_size=T, feature_size=EMBED_DIM, device='cpu'):
#         self.seq_len = seq_len
#         self.batch_size = batch_size
#         self.rolling_size = rolling_size
#         self.seq_features = torch.zeros((seq_len + rolling_size) * batch_size, feature_size).to(device)
#         self.seq_masks = torch.zeros(rolling_size * batch_size, 1).to(device)
#
#     def flip_seq_idx(self, idx):
#         seq_idx = (self.rolling_size - 1 - idx // self.batch_size) * self.batch_size + idx % self.batch_size  # invert index, when N=2, [509, 511] should select [3,1] index
#         return seq_idx
#
#     def fetch_seq(self, idx):
#         seq = self.seq_features
#         mask = self.seq_masks
#         # reverse_mask = None if mask is None else torch.flip(mask.view(T, N, -1), dims=[0]).view(T * N, -1)
#         results = []
#         for start in idx:
#             # indices = torch.arange(start, start + LOOK_BACK_SIZE * N, N)
#             # d = torch.index_select(seq, 0, indices)
#             d = seq[start:start + self.seq_len * self.batch_size:self.batch_size] * 1  # will make a new tensor
#             # mask_value, _ = torch.cummin(reverse_mask[start::N], dim=0)
#             # results.append(d * mask_value)
#
#             i = torch.arange(start, mask.shape[0], self.batch_size)
#             is_zero = (mask[i, 0] == 0)
#             non_zero_pos = is_zero.nonzero()
#             if non_zero_pos.shape[0] > 0:
#                 first_zero_index = non_zero_pos.min()
#                 d[first_zero_index:] = 0  # d is a new tensor, no modification to seq
#             results.append(d.unsqueeze(dim=1))
#         sub_seq = torch.cat(results, dim=1)
#         return sub_seq
#
#     def _roll_seq(self, seq, data, order=0):
#         if order == -1:
#             start = data.shape[0]
#             end = seq.shape[0]
#             seq = torch.cat((seq[start:end], data), dim=0)
#         else:
#             start = 0
#             end = seq.shape[0] - data.shape[0]
#             seq = torch.cat((data, seq[start:end]), dim=0)
#         return seq
#
#     def roll_seq_feature(self, data):
#         self.seq_features = self._roll_seq(self.seq_features, data)
#
#     def roll_seq_mask(self, data):
#         self.seq_masks = self._roll_seq(self.seq_masks, data)
#
#     def mask_last_epoch(self, last_mask):
#         seq = self.seq_features
#         if last_mask is not None:
#             mask = last_mask.view(-1, self.batch_size, 1)
#             cum_mask, _ = mask.cummin(dim=0)
#             cum_mask = cum_mask.view(-1, 1)
#             seq[self.rolling_size:self.rolling_size + cum_mask.shape[0]] *= cum_mask  # modify seq inplace


def normalize(x):
    x -= x.mean()
    x /= (x.std() + 1e-8) # prevent 0 fraction
    return x


def grey_crop_resize_batch(state):  # deal with batch observations
    states = []
    for i in state:
        array_3d = grey_crop_resize(i)
        array_4d = np.expand_dims(array_3d, axis=0)
        states.append(array_4d)
    states_array = np.vstack(states) # turn the stack into array
    return states_array # B*C*H*W

def grey_crop_resize(state): # deal with single observation
    img = Image.fromarray(state)
    grey_img = img.convert(mode='L')
    left = 0
    top = 34  # empirically chosen
    right = 160
    bottom = 194  # empirically chosen
    cropped_img = grey_img.crop((left, top, right, bottom))
    resized_img = cropped_img.resize((84, 84))
    array_2d = np.asarray(resized_img)
    array_3d = np.expand_dims(array_2d, axis=0)
    return array_3d / 255. # C*H*W


def compute_gae(next_value, rewards, masks, values, gamma=G_GAE, lam=L_GAE):
    values = values + [next_value]  # concat last value to the list
    gae = 0  # first gae always to 0
    returns = []

    for step in reversed(range(len(rewards))):  # for each positions with respect to the result of the action
        delta = rewards[step] + gamma * values[step + 1] * masks[step] - values[step]  # compute delta, sum of current reward and the expected goodness of the next state (next state val minus current state val), zero if 'done' is reached, so i can't consider next val
        gae = delta + gamma * lam * masks[step] * gae  # recursively compute the sum of the gae until last state is reached, gae is computed summing all gae of previous actions, higher is multiple good actions succeds, lower otherwhise
        returns.insert(0, gae + values[step])  # sum again the value of current action, so a state is better to state in if next increment as well
    return returns


def ppo_iter(states, actions, log_probs, returns, advantages, seq):
    # batch_size = states.size(0)  # lenght of data collected
    batch_size = actions.size(0) * actions.size(1)

    for _ in range(batch_size // M):
        # rand_ids = np.random.randint(0, batch_size, M)  # integer array of random indices for selecting M mini batches
        # reverse_ids = seq.flip_seq_idx(rand_ids)
        # (Len, Batch, ...)
        rand_ids_0 = np.random.randint(0, actions.size(0), M)
        rand_ids_1 = np.random.randint(0, actions.size(1), M)
        reverse_ids_0 = actions.size(0) - rand_ids_0 # one more item: 0->256, not 255
        seq_feature, seq_mask = seq.fetch_seq(reverse_ids_0, rand_ids_1)
        yield states[rand_ids_0, rand_ids_1, :], \
              actions[rand_ids_0, rand_ids_1, :], \
              log_probs[rand_ids_0, rand_ids_1, :], \
              returns[rand_ids_0, rand_ids_1, :], \
              advantages[rand_ids_0, rand_ids_1, :], \
              states[rand_ids_0 + 1, rand_ids_1, :], \
              seq_feature, \
              seq_mask


def recalc_seq(model, states, last_states, seq_np, seq=None):
    # need grad
    if seq is None:
        seq = SeqTorch(seq_len=LOOK_BACK_SIZE, batch_size=N, rolling_size=T, feature_size=EMBED_DIM, device=states.device)
    seq.seq_masks = torch.from_numpy(seq_np.seq_masks).to(states.device)

    assert T / N == T // N
    n_seq = T // N
    _, _, c, h, w = states.shape
    feature_arr = []
    for i in range(0, T, n_seq):
        feature_t = model.feature(states[i:i + n_seq, ...].view(-1, c, h, w))
        feature_t = feature_t.view(n_seq, N, EMBED_DIM)
        feature_arr.insert(0, feature_t.flip(dims=[0]))
        # index_t = torch.arange(T - i - 1, T - i - n_seq - 1, -1)
        # seq.seq_features[index_t] = feature_t
    if last_states is not None:
        feature_t = model.feature(last_states[-LOOK_BACK_SIZE:].view(-1, c, h, w))
        feature_t = feature_t.view(LOOK_BACK_SIZE, N, EMBED_DIM)
        feature_arr.append(feature_t.flip(dims=[0]))
        # index_t = torch.arange(-1, -LOOK_BACK_SIZE, -1)
        # seq.seq_features[index_t] = feature_t
    else:
        feature_arr.append(torch.from_numpy(seq_np.seq_features[-LOOK_BACK_SIZE:]).to(states.device))
        # feature_t = model.feature(torch.zeros([LOOK_BACK_SIZE * N, c, h, w], dtype=torch.float32).to(states.device))
        # feature_t = feature_t.view(LOOK_BACK_SIZE, N, EMBED_DIM)
        # feature_arr.append(feature_t)
    seq.seq_features = torch.cat(feature_arr, dim=0)
    return seq


def ppo_update(model, optimizer, states, actions, log_probs, returns, advantages, seq_np, last_states, clip_param=E_CLIP):
    params_feature = model.get_parameter('feature.linear.weight')
    seq = recalc_seq(model, states, last_states, seq_np)
    for _ in range(K):
        for state, action, old_log_probs, return_, advantage, next_state, seq_feature, seq_mask in ppo_iter(states, actions, log_probs, returns, advantages, seq):
            dist, value, _, attn_out = model(state, seq_feature, seq_mask)
            pred_feature_next_logits, feature_next = model.pred_next_feature(attn_out, action, next_state)
            # x_recon_logit = model.pred_next_state(attn_out, action)

            action = action.reshape(1, len(action)) # take the relative action and take the column
            new_log_probs = dist.log_prob(action)
            new_log_probs = new_log_probs.reshape(len(old_log_probs), 1) # take the column
            ratio = (new_log_probs - old_log_probs).exp() # new_prob/old_prob
            surr1 = ratio * advantage
            surr2 = torch.clamp(ratio, 1.0 - clip_param, 1.0 + clip_param) * advantage
            actor_loss = - torch.min(surr1, surr2).mean()
            critic_loss = (return_ - value).pow(2).mean()
            entropy_loss = -dist.entropy().mean()

            feature_next = feature_next.detach()
            # pred_loss = F.mse_loss(pred_feature_next_logits, feature_next, reduction='none').sum(dim=1).mean()

            # note: cross_entropy and binary_cross_entropy takes logits as input, not softmax/sigmoid result as input
            # cross_entropy return shape: (B, )
            # pred_loss = F.cross_entropy(pred_feature_next_logits, F.softmax(feature_next, dim=-1), reduction='none').mean()

            # pred_loss = F.binary_cross_entropy_with_logits(x_recon_logit, next_state, reduction='none').sum(dim=(1,2,3)).mean()
            pred_loss = F.binary_cross_entropy_with_logits(pred_feature_next_logits, feature_next, reduction='none').sum(dim=-1).mean()

            loss = C_1 * critic_loss + actor_loss + C_2 * entropy_loss + C_3 * pred_loss

            optimizer.zero_grad() # in PyTorch, we need to set the gradients to zero before starting to do backpropragation because PyTorch accumulates the gradients on subsequent backward passes.
            (C_1 * critic_loss).backward(retain_graph=True)
            grad_critic = params_feature.grad.mean(), params_feature.grad.std()

            optimizer.zero_grad()
            actor_loss.backward(retain_graph=True)
            grad_actor = params_feature.grad.mean(), params_feature.grad.std()

            optimizer.zero_grad()
            (- C_2 * entropy_loss).backward(retain_graph=True)
            grad_entropy = params_feature.grad.mean(), params_feature.grad.std()

            optimizer.zero_grad()
            (C_3 * pred_loss).backward(retain_graph=True)
            grad_pred = params_feature.grad.mean(), params_feature.grad.std()

            optimizer.zero_grad() # in PyTorch, we need to set the gradients to zero before starting to do backpropragation because PyTorch accumulates the gradients on subsequent backward passes.
            loss.backward() # computes dloss/dx for every parameter x which has requires_grad=True. These are accumulated into x.grad for every parameter x

            grad_total = params_feature.grad.mean(), params_feature.grad.std()
            grad_max = params_feature.grad.abs().max()

            optimizer.step() # performs the parameters update based on the current gradient and the update rule

            recalc_seq(model, states, last_states, seq_np, seq)

    return {'loss': loss, 'actor_loss': actor_loss, 'critic_loss': critic_loss, 'entropy_loss': entropy_loss, 'pred_loss': pred_loss,
            'grad_critic': grad_critic[0], 'grad_actor': grad_actor[0], 'grad_entropy': grad_entropy[0], 'grad_pred': grad_pred[0],
            'grad_total': grad_total[0], 'grad_max': grad_max,
            # 'state': state[0], 'next_state': next_state[0], 'recon_state': F.sigmoid(x_recon_logit[0])
            }


class SeqTorch:
    def __init__(self, seq_len=LOOK_BACK_SIZE, batch_size=N, rolling_size=T, feature_size=EMBED_DIM, device='cpu'):
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.rolling_size = rolling_size
        self.device = device
        self.seq_features = torch.zeros([seq_len + rolling_size, batch_size, feature_size], dtype=torch.float32).to(device)
        self.seq_masks = torch.zeros([seq_len + rolling_size, batch_size, 1], dtype=torch.float32).to(device)  # 1 running, 0 done

    def fetch_seq(self, idx0, idx1):
        seq = self.seq_features
        mask = self.seq_masks
        # (L, B, ...)
        sub_seq = torch.stack([seq[idx0 + i, idx1] * 1 for i in range(self.seq_len)], dim=0)
        sub_mask = torch.stack([mask[idx0 + i, idx1] * 1 for i in range(self.seq_len)], dim=0)
        cum_mask, _ = torch.cummin(sub_mask, dim=0)
        cum_mask = 1 - cum_mask
        cum_mask = cum_mask.squeeze(dim=2).T  # (batch, seq)
        cum_mask[cum_mask == 1] = float('-inf')
        return sub_seq, cum_mask

    # def _roll_seq(self, seq, data, order=0):
    #     if order == -1:
    #         seq[:-1] = seq[1:]
    #         seq[-1] = data
    #     else:
    #         seq[1:] = seq[:-1]
    #         seq[0] = data
    #
    # def roll_seq_feature(self, data):
    #     self._roll_seq(self.seq_features, data)
    #
    # def roll_seq_mask(self, data):
    #     self._roll_seq(self.seq_masks, data)


class Seq:
    def __init__(self, seq_len=LOOK_BACK_SIZE, batch_size=N, rolling_size=T, feature_size=EMBED_DIM, device='cpu'):
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.rolling_size = rolling_size
        self.device = device
        self.seq_features = np.zeros([seq_len + rolling_size, batch_size, feature_size], dtype=np.float32)
        self.seq_masks = np.zeros([seq_len + rolling_size, batch_size, 1], dtype=np.float32)  # 1 running, 0 done

    def fetch_seq(self, idx0, idx1):
        seq = self.seq_features
        mask = self.seq_masks
        # (L, B, ...)
        sub_seq = np.stack([seq[idx0 + i, idx1] * 1 for i in range(self.seq_len)], axis=0)
        sub_mask = np.stack([mask[idx0 + i, idx1] * 1 for i in range(self.seq_len)], axis=0)
        cum_mask = 1 - np.minimum.accumulate(sub_mask, axis=0)
        cum_mask = cum_mask.squeeze(axis=2).T  # (batch, seq)
        cum_mask[cum_mask == 1] = float('-inf')
        return torch.from_numpy(sub_seq).to(self.device), torch.from_numpy(cum_mask).to(self.device)

    def _roll_seq(self, seq, data, order=0):
        if order == -1:
            seq[:-1] = seq[1:]
            seq[-1] = data
        else:
            seq[1:] = seq[:-1]
            seq[0] = data

    def roll_seq_feature(self, data):
        self._roll_seq(self.seq_features, data)

    def roll_seq_mask(self, data):
        self._roll_seq(self.seq_masks, data)

    # def mask_last_epoch(self, last_mask):
    #     seq = self.seq_features
    #     if last_mask is not None:
    #         mask = last_mask.reshape(-1, self.batch_size, 1)
    #         cum_mask = np.minimum.accumulate(mask, axis=0)
    #         cum_mask = cum_mask.reshape(-1, 1)
    #         offset = self.rolling_size * self.batch_size
    #         size = min(seq.shape[0] - offset, cum_mask.shape[0])  # when look back size < rolling size
    #         seq[offset:offset + size] *= cum_mask[:size]  # modify seq inplace


def hook_fn(module, input, output):
    module.feature_map = output


def ppo_train(model, envs, device, optimizer, test_rewards, test_epochs, train_epoch, best_reward, early_stop=False):
    conv_models = []
    num_convs = 3
    for i in range(1, num_convs+1):
        conv = model.feature.get_submodule(f'conv{i}')
        conv.register_forward_hook(hook_fn)
        conv_models.append(conv)

    writer = SummaryWriter(comment=f'.{MODEL}.{ENV_ID}')
    env_test = gym.make(ENV_ID, render_mode='rgb_array')

    state = envs.reset()
    state = grey_crop_resize_batch(state)

    total_reward_1_env = 0
    total_runs_1_env = 0
    steps_1_env = 0

    seq = Seq(seq_len=LOOK_BACK_SIZE, batch_size=N, rolling_size=T, feature_size=EMBED_DIM, device=device)
    fixed_idx_0 = np.asarray([0 for j in range(N)])
    fixed_idx_1 = np.asarray([j for j in range(N)])

    last_states = None

    while not early_stop:
        log_probs = []
        values = []
        states = []
        actions = []
        rewards = []
        masks = []

        for i in range(T):
            state = torch.FloatTensor(state).to(device)
            dist, value, feature, _ = model(state, *seq.fetch_seq(fixed_idx_0, fixed_idx_1))
            action = dist.sample().to(device)
            next_state, reward, done, _ = envs.step(action.cpu().numpy())
            next_state = grey_crop_resize_batch(next_state)  # simplify perceptions (grayscale-> crop-> resize) to train CNN
            log_prob = dist.log_prob(action)  # needed to compute probability ratio r(theta) that prevent policy to vary too much probability related to each action (make the computations more robust)
            log_prob_vect = log_prob.reshape(len(log_prob), 1)  # transpose from row to column
            log_probs.append(log_prob_vect)
            action_vect = action.reshape(len(action), 1)  # transpose from row to column
            actions.append(action_vect)
            values.append(value)
            rewards.append(torch.FloatTensor(reward).unsqueeze(1).to(device))
            masks.append(torch.FloatTensor(1 - done).unsqueeze(1).to(device))
            states.append(state)
            state = next_state
            seq.roll_seq_feature(feature.detach().cpu().numpy())
            seq.roll_seq_mask(1 - done[:, np.newaxis])

            total_reward_1_env += reward[0]
            steps_1_env += 1
            if done[0]:
                total_runs_1_env += 1
                print(f'Run {total_runs_1_env}, steps {steps_1_env}, Reward {total_reward_1_env}')
                writer.add_scalar('Reward/train_reward_1_env', total_reward_1_env, train_epoch * T + i + 1)
                total_reward_1_env = 0
                steps_1_env = 0

        next_state = torch.FloatTensor(next_state).to(device)  # consider last state of the collection step
        _, next_value, _, _ = model(next_state, *seq.fetch_seq(fixed_idx_0, fixed_idx_1))  # collect last value effect of the last collection step
        returns = compute_gae(next_value, rewards, masks, values)

        states.append(next_state)
        returns = torch.stack(returns).detach()  # concatenates along existing dimension and detach the tensor from the network graph, making the tensor no gradient
        log_probs = torch.stack(log_probs).detach()
        values = torch.stack(values).detach()
        states = torch.stack(states)
        actions = torch.stack(actions)
        advantages = returns - values  # compute advantage for each action
        advantages = normalize(advantages)  # compute the normalization of the vector to make uniform values
        results = ppo_update(model, optimizer, states, actions, log_probs, returns, advantages, seq, last_states)
        last_states = states[-LOOK_BACK_SIZE:]
        train_epoch += 1

        total_steps = train_epoch * T
        writer.add_scalar('Loss/Total Loss', results['loss'].item(), total_steps)
        writer.add_scalar('Loss/Actor Loss', results['actor_loss'].item(), total_steps)
        writer.add_scalar('Loss/Critic Loss', results['critic_loss'].item(), total_steps)
        writer.add_scalar('Loss/Entropy Loss', results['entropy_loss'].item(), total_steps)
        writer.add_scalar('Loss/Predict Loss', results['pred_loss'].item(), total_steps)

        writer.add_scalar('Grad/Max', results['grad_max'].item(), total_steps)
        writer.add_scalar('Grad/Critic', results['grad_critic'].item(), total_steps)
        writer.add_scalar('Grad/Actor', results['grad_actor'].item(), total_steps)
        writer.add_scalar('Grad/Entropy', results['grad_entropy'].item(), total_steps)
        writer.add_scalar('Grad/Pred', results['grad_pred'].item(), total_steps)
        writer.add_scalar('Grad/Total', results['grad_total'].item(), total_steps)

        if train_epoch % T_EPOCHS == 0:  # do a test every T_EPOCHS times
            # writer.add_image('Image/state', results['state'].detach().cpu().numpy(), total_steps)
            # writer.add_image('Image/next_state', results['next_state'].detach().cpu().numpy(), total_steps)
            # writer.add_image('Image/recon_state', results['recon_state'].detach().cpu().numpy(), total_steps)

            for i in range(num_convs):
                b, c, h, w = conv_models[i].feature_map.shape
                feature_map = conv_models[i].feature_map[:16].transpose(0, 1).unsqueeze(dim=2).reshape(-1, 1, h, w)
                feature_map_grids = torchvision.utils.make_grid(feature_map, nrow=16, padding=1, normalize=True)
                writer.add_image(f"FeatureMap/conv{i + 1}", feature_map_grids.detach().cpu().numpy(), total_steps)

            model.eval()
            test_reward = np.mean([test_env(env_test, model, device) for _ in range(N_TESTS)])  # do N_TESTS tests and takes the mean reward
            model.train()
            test_rewards.append(test_reward)  # collect the mean rewards for saving performance metric
            test_epochs.append(train_epoch)
            print('Epoch: %s -> Reward: %s' % (train_epoch, test_reward))
            writer.add_scalar('Reward/test_reward', test_reward, total_steps)

            if best_reward is None or best_reward < test_reward:  # save a checkpoint every time it achieves a better reward
                if best_reward is not None:
                    print("Best reward updated: %.3f -> %.3f" % (best_reward, test_reward))
                    name = "%s_%s_%+.3f_%d.pth" % (MODEL, ENV_ID, test_reward, train_epoch)
                    fname = os.path.join(MODEL_DIR, name)
                    states = {
                        'state_dict': model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'test_rewards': test_rewards,
                        'test_epochs': test_epochs,
                    }
                    torch.save(states, fname)

                best_reward = test_reward

            if test_reward > TARGET_REWARD:  # stop training if archive the best
                early_stop = True


def train(load_from=None):
    print('Env:', ENV_ID)
    print('Model:', MODEL)
    use_cuda = torch.cuda.is_available()  # Autodetect CUDA
    device = torch.device("cuda" if use_cuda else "cpu")
    print('Device:', device)

    envs = [lambda: gym.make(ENV_ID, render_mode='rgb_array')] * N  # Prepare N actors in N environments
    envs = SubprocVecEnv(envs)  # Vectorized Environments are a method for stacking multiple independent environments into a single environment. Instead of the training an RL agent on 1 environment per step, it allows us to train it on n environments per step. Because of this, actions passed to the environment are now a vector (of dimension n). It is the same for observations, rewards and end of episode signals (dones). In the case of non-array observation spaces such as Dict or Tuple, where different sub-spaces may have different shapes, the sub-observations are vectors (of dimension n).
    num_inputs = 1
    num_outputs = envs.action_space.n
    model = ActorCritic(num_inputs, num_outputs, H_SIZE).to(device)
    optimizer = optim.Adam(model.parameters(), lr=L_RATE)  # implements Adam algorithm
    test_rewards = []
    test_epochs = []
    train_epoch = 0
    best_reward = None

    summary(model, input_size=[(1, 84, 84), (1, H_SIZE)], batch_dim=0, dtypes=[torch.float, torch.float])

    if load_from is not None:
        checkpoint = torch.load(load_from, map_location=None if use_cuda else torch.device('cpu'))
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        test_rewards = checkpoint['test_rewards']
        test_epochs = checkpoint['test_epochs']
        train_epoch = test_epochs[-1]
        best_reward = test_rewards[-1]
        print(f'Model loaded, starting from epoch %d' % (train_epoch + 1))
        print('Previous best reward: %.3f' % (best_reward))

    print(model)
    print(optimizer)

    ppo_train(model, envs, device, optimizer, test_rewards, test_epochs, train_epoch, best_reward)


def test_env(env, model, device):
    state, _ = env.reset()
    state = grey_crop_resize(state)

    done = False
    total_reward = 0

    # set rolling_size=0
    seq = Seq(seq_len=LOOK_BACK_SIZE, batch_size=1, rolling_size=0, feature_size=EMBED_DIM, device=device)
    fixed_idx_0 = np.asarray([0 for j in range(1)])
    fixed_idx_1 = np.asarray([j for j in range(1)])

    while not done:
        state = torch.FloatTensor(state).unsqueeze(0).to(device)
        dist, _, feature, _ = model(state, *seq.fetch_seq(fixed_idx_0, fixed_idx_1))
        action = dist.sample().cpu().numpy()[0]
        next_state, reward, done, _, _ = env.step(action)
        next_state = grey_crop_resize(next_state)
        state = next_state
        total_reward += reward

        done = np.asarray([done])
        mask = 1 - done[:, np.newaxis]
        seq.roll_seq_feature(feature.detach().cpu().numpy())
        seq.roll_seq_mask(mask)

    return total_reward


def eval(load_from):
    use_cuda = torch.cuda.is_available()  # Autodetect CUDA
    device = torch.device("cuda" if use_cuda else "cpu")
    print('Device:', device)

    env_test = gym.make(ENV_ID, render_mode='human')

    num_inputs = 1
    num_outputs = env_test.action_space.n
    model = ActorCritic(num_inputs, num_outputs, H_SIZE).to(device)
    model.eval()

    checkpoint = torch.load(load_from, map_location=None if use_cuda else torch.device('cpu'))
    model.load_state_dict(checkpoint['state_dict'])

    while True:
        test_env(env_test, model, device)


if __name__ == "__main__":
    # python ppo_pong.py --eval --model models/ppo_pong.diff.5200.pth
    ap = argparse.ArgumentParser(description='Process args.')
    ap.add_argument('--eval', action='store_true', help='evaluate')
    ap.add_argument('--model', type=str, default=None, help='model to load')
    args = ap.parse_args()

    if not args.eval:
        train(args.model)
    else:
        eval(args.model)
