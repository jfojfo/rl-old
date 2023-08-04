import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import Dataset, DataLoader
import torchvision
import numpy as np
import gymnasium as gym
import argparse
from PIL import Image


# good: https://hrl.boyuai.com/chapter/2/ppo%E7%AE%97%E6%B3%95/


# Hyperparameters
NUM_ENVS = 8
EP_MAX = 20000
EP_LEN = 256
K_EPOCHS = 10
GAMMA = 0.99
EPS_CLIP = 0.2
LR_A = 1e-4
LR_C = 1e-4
BATCH_SIZE = 64

MODEL_DIR = 'models'
MODEL = 'ppo_episode.single_round'
ENV_ID = 'PongDeterministic-v0'


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Residual block implementation
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.downsample = None
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

# Define the actor and critic networks
class ActorCritic(nn.Module):
    # def __init__(self, input_shape, output_dim):
    #     super(ActorCritic, self).__init__()
    #     # N = input_shape[0] * input_shape[1] * input_shape[2]
    #     self.conv_feature = nn.Sequential(
    #         # 3, 80, 104
    #         ResidualBlock(3, 16, stride=2),   # 16, 40, 52
    #         ResidualBlock(16, 32, stride=2),  # 32, 20, 26
    #         ResidualBlock(32, 32, stride=2),  # 32, 10, 13
    #         nn.Flatten(),
    #     )
    #     N = 32 * 10 * 13
    #     self.actor = nn.Sequential(
    #         nn.Linear(N, 128),
    #         nn.ReLU(inplace=True),
    #         nn.Linear(128, output_dim),
    #         nn.Softmax(dim=-1),
    #     )
    #     self.critic = nn.Sequential(
    #         nn.Linear(N, 128),
    #         nn.ReLU(inplace=True),
    #         nn.Linear(128, 1),
    #     )

    def __init__(self, input_shape, output_dim):
        super(ActorCritic, self).__init__()
        # self.feature = nn.Sequential(
        #     nn.Linear(N, 200),
        #     nn.ReLU(inplace=True),
        # )
        # self.feature = nn.Flatten()

        self.critic = nn.Sequential(  # The “Critic” estimates the value function
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(in_features=2592, out_features=256),
            nn.ReLU(),
            nn.Linear(in_features=256, out_features=1),
        )
        self.actor = nn.Sequential(  # The “Actor” updates the policy distribution in the direction suggested by the Critic (such as with policy gradients)
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(in_features=2592, out_features=256),
            nn.ReLU(),
            nn.Linear(in_features=256, out_features=output_dim),
            nn.Softmax(dim=1),
        )

        # N = 84 * 84
        # self.actor = nn.Sequential(
        #     nn.Flatten(),
        #     nn.Linear(N, 200),
        #     nn.ReLU(),
        #     nn.Linear(200, output_dim),
        #     nn.Softmax(dim=-1),
        # )
        # self.critic = nn.Sequential(
        #     nn.Flatten(),
        #     nn.Linear(N, 200),
        #     nn.ReLU(),
        #     nn.Linear(200, 1),
        # )

    # def forward(self, x):
        # action_probs = torch.softmax(self.actor(x), dim=-1)
        # action_probs = self.actor(x)
        # state_values = self.critic(x)
        # return action_probs, state_values

    def act(self, state):
        # feature = self.feature(state)
        feature = state
        action_probs = self.actor(feature)
        dist = torch.distributions.Categorical(action_probs)
        action = dist.sample()
        action_logprob = dist.log_prob(action)
        state_val = self.critic(feature).squeeze(-1)
        return action.detach(), action_logprob.detach(), state_val.detach()

    def evaluate(self, states, actions):
        # feature = self.feature(states)
        feature = states
        action_probs = self.actor(feature)
        dist = torch.distributions.Categorical(action_probs)
        action_logprobs = dist.log_prob(actions)
        dist_entropy = dist.entropy()
        state_values = self.critic(feature)
        return action_logprobs, state_values, dist_entropy

class RolloutBuffer(Dataset):
    def __init__(self, batch_size=32, transform=None):
        self.transform = transform
        self.batch_size = batch_size
        self.states = []
        self.actions = []
        self.logprobs = []
        self.rewards = []
        self.state_values = []
        self.dones = []
        self.d_rewards = []

    def __len__(self):
        return len(self.rewards)

    def __getitem__(self, index):
        old_state = self.states[index]
        old_action = self.actions[index]
        old_logprob = self.logprobs[index]
        old_state_value = self.state_values[index]
        d_reward = self.d_rewards[index]
        return old_state, old_action, old_logprob, old_state_value, d_reward

    def make_batch(self):
        data_loader = DataLoader(self, batch_size=self.batch_size, shuffle=True, drop_last=True)
        return data_loader

    def clear(self):
        del self.states[:]
        del self.actions[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.state_values[:]
        del self.dones[:]
        del self.d_rewards[:]


# Define the PPO algorithm
class PPO:
    def __init__(self, input_dim, output_dim, lr_a=0.0001, lr_c=0.0001, gamma=0.99, K_epochs=4, eps_clip=0.2, batch_size=256):
        self.gamma = gamma
        self.K_epochs = K_epochs
        self.eps_clip = eps_clip
        self.buffer = RolloutBuffer(batch_size=batch_size)

        self.policy = ActorCritic(input_dim, output_dim)
        # self.optimizer = optim.Adam([
        #     {'params': self.policy.actor.parameters(), 'lr': lr_a},
        #     {'params': self.policy.critic.parameters(), 'lr': lr_c},
        # ])
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr_c)
        # self.policy_old = ActorCritic(input_dim, output_dim)
        self.policy_old = self.policy
        # self.policy_old.load_state_dict(self.policy.state_dict())  # Copy initial weights

    def select_action(self, state, store=True):
        with torch.no_grad():
            state = torch.FloatTensor(state).unsqueeze(0).to(device)
            action, logprob, state_value = self.policy_old.act(state)
        if store:
            self.buffer.states.append(state.cpu().squeeze(0).detach())
            self.buffer.actions.append(action.cpu().squeeze(0).detach())
            self.buffer.logprobs.append(logprob.cpu().squeeze(0).detach())
            self.buffer.state_values.append(state_value.cpu().squeeze(0).detach())
        return action.cpu().item()

    def add_buffer(self, reward, done):
        self.buffer.rewards.append(reward)
        self.buffer.dones.append(done)

    def compute_value(self, state):
        with torch.no_grad():
            state = torch.FloatTensor(state).unsqueeze(0).to(device)
            action, logprob, state_value = self.policy_old.act(state)
        return state_value.cpu().item()

    def discounted_rewards(self):
        values = self.buffer.state_values + [0.0]
        returns = []
        running_returns = 0
        gae = 0
        for t in reversed(range(len(self.buffer.rewards))):
            # if self.buffer.rewards[t] in (-1, 1):
            #     running_returns = 0
            # mask = 1.0 - self.buffer.dones[t]
            mask = 1
            if self.buffer.dones[t] or self.buffer.rewards[t] in (-1, 1):
                mask = 0
            delta = self.buffer.rewards[t] + self.gamma * values[t + 1] * mask - values[t]
            gae = delta + self.gamma * 0.95 * gae * mask
            returns.append(gae + values[t])
        returns.reverse()
        return [torch.tensor(item) for item in returns]

    def update(self):
        # self.policy_old.load_state_dict(self.policy.state_dict())

        d_rewards = self.discounted_rewards()
        self.buffer.d_rewards = d_rewards

        # PPO policy updates
        for _ in range(self.K_epochs):
            epoch_loss = torch.tensor(0.0).to(device)
            epoch_actor_loss = torch.tensor(0.0).to(device)
            epoch_critic_loss = torch.tensor(0.0).to(device)
            epoch_entropy_loss = torch.tensor(0.0).to(device)

            data_loader = self.buffer.make_batch()
            for batch in data_loader:
                batch = [item.to(device) for item in batch]
                if not data_loader.drop_last and len(batch[0]) == 1:
                    # advantage.std() will be nan
                    break
                loss, actor_loss, critic_loss, entropy_loss = self.loss_single_batch(*batch)
                epoch_loss += loss
                epoch_actor_loss += actor_loss
                epoch_critic_loss += critic_loss
                epoch_entropy_loss += entropy_loss

                self.optimizer.zero_grad()
                (loss / batch[0].shape[0]).backward()
                self.optimizer.step()

            epoch_loss = epoch_loss / len(data_loader.dataset)
            epoch_actor_loss = epoch_actor_loss / len(data_loader.dataset)
            epoch_critic_loss = epoch_critic_loss / len(data_loader.dataset)
            epoch_entropy_loss = epoch_entropy_loss / len(data_loader.dataset)

        return epoch_loss, epoch_actor_loss, epoch_critic_loss, epoch_entropy_loss

    def loss_single_batch(self, old_states, old_actions, old_logprobs, old_state_values, d_rewards):
        # Calculate surrogate loss
        logprobs, state_values, dist_entropy = self.policy.evaluate(old_states, old_actions)
        state_values = torch.squeeze(state_values, -1)
        ratios = torch.exp(logprobs - old_logprobs.detach())

        # Calculate advantage
        advantages = (d_rewards - state_values).detach()
        # advantages_norm = (advantages - advantages.mean()) / (advantages.std() + 1e-7)

        surr1 = ratios * advantages
        surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages

        actor_loss = -torch.min(surr1, surr2).sum()
        # critic_loss = F.mse_loss(state_values, d_rewards, reduction="sum")
        critic_loss = 0.5 * (state_values - d_rewards).pow(2).sum()
        entropy_loss = -0.01 * dist_entropy.sum()
        loss = actor_loss + critic_loss + entropy_loss
        return loss, actor_loss, critic_loss, entropy_loss

# def preprocess(image):
#     img = image[1:-1, :, :]
#     scale = torchvision.transforms.ToTensor()
#     img = scale(img).transpose(1, 2)
#     img = torchvision.transforms.functional.resize(img, (int(img.shape[1] / 2), int(img.shape[2] / 2)))
#     return img
def preprocess(prev, curr):
    img = curr[35:195]
    img = img[::2, ::2, 0]
    img[img == 144] = 0
    img[img == 109] = 0
    img[img != 0] = 1
    if prev is None:
        return img.astype(np.float32).ravel()
    x2 = img.astype(np.float32)

    img = prev[35:195]
    img = img[::2, ::2, 0]
    img[img == 144] = 0
    img[img == 109] = 0
    img[img != 0] = 1
    x1 = img.astype(np.float32)

    diff = ((x2 - x1) + 1.0) / 2.0
    return diff

def grey_crop_resize(state): # deal with single observation
    img = Image.fromarray(state)
    grey_img = img.convert(mode='L')
    left = 0
    top = 35  # empirically chosen
    right = 160
    bottom = 195  # empirically chosen
    cropped_img = grey_img.crop((left, top, right, bottom))
    resized_img = cropped_img.resize((84, 84))
    array_2d = np.asarray(resized_img)
    array_3d = np.expand_dims(array_2d, axis=0)
    return array_3d / 255. # C*H*W


# Training loop
def train(num_epochs, load_from=None, eval=False):
    render_mode = 'human' if eval else 'rgb_array'
    # PongDeterministic-v0
    env = gym.make(ENV_ID, render_mode=render_mode)
    input_shape = env.observation_space.shape
    output_dim = env.action_space.n
    ppo_agent = PPO(input_shape, output_dim, lr_a=LR_A, lr_c=LR_C, gamma=GAMMA, K_epochs=K_EPOCHS, eps_clip=EPS_CLIP, batch_size=BATCH_SIZE)
    ppo_agent.policy.to(device)
    ppo_agent.policy_old.to(device)

    if eval:
        ppo_agent.policy.eval()
        ppo_agent.policy_old.eval()

    writer = SummaryWriter(comment=f'.{MODEL}.{ENV_ID}')

    start_epoch = 0
    total_steps = 0

    if load_from is not None:
        saved_dict = torch.load(load_from, map_location=torch.device('cpu') if eval else None)
        ppo_agent.policy.load_state_dict(saved_dict['state_dict'])
        ppo_agent.policy_old.load_state_dict(saved_dict['state_dict'])
        ppo_agent.optimizer.load_state_dict(saved_dict['optimizer_state_dict'])
        start_epoch = saved_dict['epoch']
        total_steps = saved_dict['total_steps']
        print(f"loaded model from epoch {start_epoch}...")

    for epoch in range(start_epoch, num_epochs):
        curr_frame, _ = env.reset()
        # state = preprocess(None, curr_frame)
        state = grey_crop_resize(curr_frame)
        total_reward = 0
        step = 0
        ppo_agent.buffer.clear()
        while step < EP_MAX:
            step += 1
            total_steps += 1

            action = ppo_agent.select_action(state, store=not eval)
            next_frame, reward, done, _, _ = env.step(action)
            # next_state = preprocess(curr_frame, next_frame)
            next_state = grey_crop_resize(next_frame)

            if not eval:
                ppo_agent.add_buffer(reward, done)

            curr_frame = next_frame
            state = next_state
            total_reward += reward

            # if step % EP_LEN == 0 or step == EP_MAX or done:
            if done and not eval:
                if len(ppo_agent.buffer.rewards) > 0:
                    loss, actor_loss, critic_loss, entropy_loss = ppo_agent.update()
                    ppo_agent.buffer.clear()

                    writer.add_scalar('Loss/Total Loss', loss.item(), total_steps)
                    writer.add_scalar('Loss/Actor Loss', actor_loss.item(), total_steps)
                    writer.add_scalar('Loss/Critic Loss', critic_loss.item(), total_steps)
                    writer.add_scalar('Loss/Entropy Loss', entropy_loss.item(), total_steps)

            if done:
                break

        print(f"Epoch {epoch+1}, steps: {step}, Total Reward: {total_reward}")
        writer.add_scalar('Reward', total_reward, total_steps)

        if (epoch + 1) % 100 == 0 and not eval:
            torch.save({
                'state_dict': ppo_agent.policy.state_dict(),
                'optimizer_state_dict': ppo_agent.optimizer.state_dict(),
                'epoch': epoch + 1,
                'total_steps': total_steps,
            }, f'{MODEL_DIR}/{MODEL}.{epoch + 1}.pth')

    env.close()

# Run the training loop
if __name__ == "__main__":
    # python ppo_pong.py --eval --model models/ppo_pong.diff.5200.pth
    ap = argparse.ArgumentParser(description='Process args.')
    ap.add_argument('--eval', action='store_true', help='evaluate')
    ap.add_argument('--model', type=str, default=None, help='model to load')
    args = ap.parse_args()

    train(50000, args.model, args.eval)
