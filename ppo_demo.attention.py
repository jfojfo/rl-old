import os
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
import time


#
# https://github.com/rossettisimone/PPO_PONG_DISCRETE/blob/master/PPO_PONG.ipynb
#


H_SIZE = 128 # hidden size, linear units of the output layer
L_RATE = 1e-4 # learning rate, gradient coefficient for CNN weight update
G_GAE = 0.99 # gamma param for GAE
L_GAE = 0.95 # lambda param for GAE
E_CLIP = 0.2 # clipping coefficient
C_1 = 0.5 # squared loss coefficient
C_2 = 0.01 # entropy coefficient
C_3 = 0.1 # image reconstruction loss coefficient
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
MODEL = 'ppo_demo.attention_test'
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
        src2, w = self.self_attn(query, key, value, attn_mask=src_mask, key_padding_mask=src_key_padding_mask, )  # 计算多头注意力
        # src2: [src_len,batch_size,num_heads*kdim] num_heads*kdim = embed_dim
        src = src + self.dropout1(src2)  # 残差连接
        src = self.norm1(src)  # [src_len,batch_size,num_heads*kdim]

        src2 = self.activation(self.linear1(src))  # [src_len,batch_size,dim_feedforward]
        src2 = self.linear2(self.dropout(src2))  # [src_len,batch_size,num_heads*kdim]
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src, w  # [src_len, batch_size, num_heads * kdim] <==> [src_len,batch_size,embed_dim]


class ActorCritic(nn.Module):
    def __init__(self, num_inputs, num_outputs, hidden_size=H_SIZE, embed_dim=EMBED_DIM):
        super(ActorCritic, self).__init__()
        self.feature = nn.Sequential(
            nn.Conv2d(in_channels=num_inputs, out_channels=16, kernel_size=8, stride=4),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=4, stride=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(in_features=2592, out_features=hidden_size),
            nn.ReLU(),
        )
        # self.attention = nn.MultiheadAttention(embed_dim, 16)
        self.attention = MyTransformerEncoderLayer(embed_dim, 16, 256)

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
        attention, scores = self.attention(query, memory, memory, src_key_padding_mask=mask)
        attn_out = attention.squeeze(dim=0)
        value = self.critic(attn_out)
        probs = self.actor(attn_out)
        dist = Categorical(probs)
        return dist, value, feature


class SeqTorch:
    def __init__(self, seq_len=LOOK_BACK_SIZE, batch_size=N, rolling_size=T, feature_size=EMBED_DIM, device='cpu'):
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.rolling_size = rolling_size
        self.seq_features = torch.zeros((seq_len + rolling_size) * batch_size, feature_size).to(device)
        self.seq_masks = torch.zeros(rolling_size * batch_size, 1).to(device)

    def flip_seq_idx(self, idx):
        seq_idx = (self.rolling_size - 1 - idx // self.batch_size) * self.batch_size + idx % self.batch_size  # invert index, when N=2, [509, 511] should select [3,1] index
        return seq_idx

    def fetch_seq(self, idx):
        seq = self.seq_features
        mask = self.seq_masks
        # reverse_mask = None if mask is None else torch.flip(mask.view(T, N, -1), dims=[0]).view(T * N, -1)
        results = []
        for start in idx:
            # indices = torch.arange(start, start + LOOK_BACK_SIZE * N, N)
            # d = torch.index_select(seq, 0, indices)
            d = seq[start:start + self.seq_len * self.batch_size:self.batch_size] * 1  # will make a new tensor
            # mask_value, _ = torch.cummin(reverse_mask[start::N], dim=0)
            # results.append(d * mask_value)

            i = torch.arange(start, mask.shape[0], self.batch_size)
            is_zero = (mask[i, 0] == 0)
            non_zero_pos = is_zero.nonzero()
            if non_zero_pos.shape[0] > 0:
                first_zero_index = non_zero_pos.min()
                d[first_zero_index:] = 0  # d is a new tensor, no modification to seq
            results.append(d.unsqueeze(dim=1))
        sub_seq = torch.cat(results, dim=1)
        return sub_seq

    def _roll_seq(self, seq, data, order=0):
        if order == -1:
            start = data.shape[0]
            end = seq.shape[0]
            seq = torch.cat((seq[start:end], data), dim=0)
        else:
            start = 0
            end = seq.shape[0] - data.shape[0]
            seq = torch.cat((data, seq[start:end]), dim=0)
        return seq

    def roll_seq_feature(self, data):
        self.seq_features = self._roll_seq(self.seq_features, data)

    def roll_seq_mask(self, data):
        self.seq_masks = self._roll_seq(self.seq_masks, data)

    def mask_last_epoch(self, last_mask):
        seq = self.seq_features
        if last_mask is not None:
            mask = last_mask.view(-1, self.batch_size, 1)
            cum_mask, _ = mask.cummin(dim=0)
            cum_mask = cum_mask.view(-1, 1)
            seq[self.rolling_size:self.rolling_size + cum_mask.shape[0]] *= cum_mask  # modify seq inplace


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
    batch_size = states.size(0)  # lenght of data collected

    for _ in range(batch_size // M):
        rand_ids = np.random.randint(0, batch_size, M)  # integer array of random indices for selecting M mini batches
        reverse_ids = seq.flip_seq_idx(rand_ids)
        seq_feature, seq_mask = seq.fetch_seq(reverse_ids)
        yield states[rand_ids, :], actions[rand_ids, :], log_probs[rand_ids, :], returns[rand_ids, :], advantages[rand_ids, :], seq_feature, seq_mask


def ppo_update(model, optimizer, states, actions, log_probs, returns, advantages, seq, clip_param=E_CLIP):
    for _ in range(K):
        for state, action, old_log_probs, return_, advantage, seq_feature, seq_mask in ppo_iter(states, actions, log_probs, returns, advantages, seq):
            dist, value, _ = model(state, seq_feature, seq_mask)
            action = action.reshape(1, len(action)) # take the relative action and take the column
            new_log_probs = dist.log_prob(action)
            new_log_probs = new_log_probs.reshape(len(old_log_probs), 1) # take the column
            ratio = (new_log_probs - old_log_probs).exp() # new_prob/old_prob
            surr1 = ratio * advantage
            surr2 = torch.clamp(ratio, 1.0 - clip_param, 1.0 + clip_param) * advantage
            actor_loss = - torch.min(surr1, surr2).mean()
            critic_loss = (return_ - value).pow(2).mean()
            entropy_loss = -dist.entropy().mean()
            loss = C_1 * critic_loss + actor_loss + C_2 * entropy_loss # loss function clip+vs+f
            optimizer.zero_grad() # in PyTorch, we need to set the gradients to zero before starting to do backpropragation because PyTorch accumulates the gradients on subsequent backward passes.
            loss.backward() # computes dloss/dx for every parameter x which has requires_grad=True. These are accumulated into x.grad for every parameter x
            optimizer.step() # performs the parameters update based on the current gradient and the update rule
    return loss, actor_loss, critic_loss, entropy_loss


class Seq:
    def __init__(self, seq_len=LOOK_BACK_SIZE, batch_size=N, rolling_size=T, feature_size=EMBED_DIM, device='cpu'):
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.rolling_size = rolling_size
        self.device = device
        self.seq_features = np.zeros([(seq_len + rolling_size) * batch_size, feature_size], dtype=np.float32)
        self.seq_masks = np.zeros([(seq_len + rolling_size) * batch_size, 1], dtype=np.float32)

    def flip_seq_idx(self, idx):
        seq_idx = (self.rolling_size - 1 - idx // self.batch_size) * self.batch_size + idx % self.batch_size  # invert index, when N=2, [509, 511] should select [3,1] index
        return seq_idx

    def fetch_seq(self, idx):
        seq = self.seq_features
        mask = self.seq_masks
        d_list = []
        m_list = []
        for start in idx:
            d = seq[start:start + self.seq_len * self.batch_size:self.batch_size] * 1  # will make a new tensor
            m = mask[start:start + self.seq_len * self.batch_size:self.batch_size] * 1  # will make a new tensor
            d_list.append(d[:, np.newaxis, :])
            m_list.append(m[:, np.newaxis, :])
        sub_seq = np.concatenate(d_list, axis=1)
        sub_mask = np.concatenate(m_list, axis=1)
        cum_mask = 1 - np.minimum.accumulate(sub_mask, axis=0)
        cum_mask = cum_mask.squeeze(axis=2).T  # (batch, seq)
        cum_mask[cum_mask == 1] = -1e-7
        return torch.from_numpy(sub_seq).to(self.device), torch.from_numpy(cum_mask).to(self.device)

    def _roll_seq(self, seq, data, order=0):
        n = data.shape[0]
        if order == -1:
            seq[:-n] = seq[n:]
            seq[-n:] = data
        else:
            seq[n:] = seq[:-n]
            seq[:n] = data

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

def ppo_train(model, envs, device, optimizer, test_rewards, test_epochs, train_epoch, best_reward, early_stop=False):
    writer = SummaryWriter(comment=f'.{MODEL}.{ENV_ID}')
    env_test = gym.make(ENV_ID, render_mode='rgb_array')

    state = envs.reset()
    state = grey_crop_resize_batch(state)

    total_reward_1_env = 0
    total_runs_1_env = 0
    steps_1_env = 0

    seq = Seq(seq_len=LOOK_BACK_SIZE, batch_size=N, rolling_size=T, feature_size=EMBED_DIM, device=device)
    fixed_idx = np.asarray([j for j in range(N)])

    while not early_stop:
        log_probs = []
        values = []
        states = []
        actions = []
        rewards = []
        masks = []

        for i in range(T):
            state = torch.FloatTensor(state).to(device)
            dist, value, feature = model(state, *seq.fetch_seq(fixed_idx))
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
        _, next_value, _ = model(next_state, *seq.fetch_seq(fixed_idx))  # collect last value effect of the last collection step
        returns = compute_gae(next_value, rewards, masks, values)
        returns = torch.cat(returns).detach()  # concatenates along existing dimension and detach the tensor from the network graph, making the tensor no gradient
        log_probs = torch.cat(log_probs).detach()
        values = torch.cat(values).detach()
        states = torch.cat(states)
        actions = torch.cat(actions)
        advantages = returns - values  # compute advantage for each action
        advantages = normalize(advantages)  # compute the normalization of the vector to make uniform values
        loss, actor_loss, critic_loss, entropy_loss = ppo_update(
            model, optimizer, states, actions, log_probs, returns, advantages, seq)
        train_epoch += 1

        total_steps = train_epoch * T
        writer.add_scalar('Loss/Total Loss', loss.item(), total_steps)
        writer.add_scalar('Loss/Actor Loss', actor_loss.item(), total_steps)
        writer.add_scalar('Loss/Critic Loss', critic_loss.item(), total_steps)
        writer.add_scalar('Loss/Entropy Loss', entropy_loss.item(), total_steps)

        if train_epoch % T_EPOCHS == 0:  # do a test every T_EPOCHS times
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
    fixed_idx = np.asarray([j for j in range(1)])

    while not done:
        state = torch.FloatTensor(state).unsqueeze(0).to(device)
        dist, _, feature = model(state, *seq.fetch_seq(fixed_idx))
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
