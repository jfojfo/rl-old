import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import Dataset, DataLoader
import torchvision
import numpy as np
import gym
import argparse


# Hyperparameters
NUM_ENVS = 8
EP_MAX = 20000
EP_LEN = 256
K_EPOCHS = 8
GAMMA = 0.99
EPS_CLIP = 0.2
LR_A = 0.0001
LR_C = 0.0001
BATCH_SIZE = 512

model_dir = 'models'
model = 'ppo_pong.diff.RMSprop'

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
        N = 80 * 80
        self.feature = nn.Sequential(
            nn.Linear(N, 200),
            nn.ReLU(inplace=True),
        )
        # self.feature = nn.Flatten()
        self.actor = nn.Sequential(
            # nn.Linear(N, 128),
            # nn.ReLU(inplace=True),
            nn.Linear(200, output_dim),
            nn.Softmax(dim=-1),
        )
        self.critic = nn.Sequential(
            # nn.Linear(N, 128),
            nn.Linear(200, 1),
        )

    # def forward(self, x):
        # action_probs = torch.softmax(self.actor(x), dim=-1)
        # action_probs = self.actor(x)
        # state_values = self.critic(x)
        # return action_probs, state_values

    def act(self, state):
        feature = self.feature(state)
        action_probs = self.actor(feature)
        dist = torch.distributions.Categorical(action_probs)
        action = dist.sample()
        action_logprob = dist.log_prob(action)
        state_val = self.critic(feature).squeeze(-1)
        return action.detach(), action_logprob.detach(), state_val.detach()

    def evaluate(self, states, actions):
        feature = self.feature(states)
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

        # Calculate advantage
        advantage = d_reward - old_state_value
        # d_rewards = (d_rewards - d_rewards.mean()) / (d_rewards.std() + 1e-7)

        return old_state, old_action, old_logprob, old_state_value, d_reward, advantage

    def make_batch(self):
        data_loader = DataLoader(self, batch_size=self.batch_size, shuffle=True)
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
        self.optimizer_a = optim.RMSprop(self.policy.parameters(), lr=lr_a)
        self.optimizer_c = optim.RMSprop(self.policy.parameters(), lr=lr_c)
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
        returns = []
        running_returns = 0
        for t in reversed(range(len(self.buffer.rewards))):
            if self.buffer.rewards[t] in (-1, 1):
                running_returns = 0
            running_returns = self.buffer.rewards[t] + self.gamma * running_returns * (1.0 - self.buffer.dones[t])
            returns.append(running_returns)
        returns.reverse()
        d_rewards = np.array(returns, dtype=np.float32)
        d_rewards = (d_rewards - np.mean(d_rewards)) / (np.std(d_rewards) + 1e-7)
        return [torch.tensor(item) for item in d_rewards]

    def update(self):
        # self.policy_old.load_state_dict(self.policy.state_dict())

        d_rewards = self.discounted_rewards()
        self.buffer.d_rewards = d_rewards

        epoch_loss = None
        epoch_actor_loss = None
        epoch_critic_loss = None
        epoch_entropy_loss = None

        # PPO policy updates
        for _ in range(self.K_epochs):
            epoch_loss = torch.tensor(0.0).to(device)
            epoch_actor_loss = torch.tensor(0.0).to(device)
            # epoch_critic_loss = torch.tensor(0.0).to(device)
            epoch_entropy_loss = torch.tensor(0.0).to(device)

            data_loader = self.buffer.make_batch()
            for batch in data_loader:
                batch = [item.to(device) for item in batch]
                loss, actor_loss, critic_loss, entropy_loss = self.loss_single_batch(*batch)
                epoch_loss += loss
                epoch_actor_loss += actor_loss
                # epoch_critic_loss += critic_loss
                epoch_entropy_loss += entropy_loss
            # Optimize the model
            epoch_loss = epoch_loss / len(data_loader.dataset)
            epoch_actor_loss = epoch_actor_loss / len(data_loader.dataset)
            # epoch_critic_loss = epoch_critic_loss / len(data_loader.dataset)
            epoch_entropy_loss = epoch_entropy_loss / len(data_loader.dataset)
            self.optimizer_a.zero_grad()
            (epoch_actor_loss + epoch_entropy_loss).backward()
            self.optimizer_a.step()

        for _ in range(self.K_epochs):
            epoch_loss = torch.tensor(0.0).to(device)
            # epoch_actor_loss = torch.tensor(0.0).to(device)
            epoch_critic_loss = torch.tensor(0.0).to(device)

            data_loader = self.buffer.make_batch()
            for batch in data_loader:
                batch = [item.to(device) for item in batch]
                loss, actor_loss, critic_loss, entropy_loss = self.loss_single_batch(*batch)
                epoch_loss += loss
                # epoch_actor_loss += actor_loss
                epoch_critic_loss += critic_loss
            # Optimize the model
            epoch_loss = epoch_loss / len(data_loader.dataset)
            # epoch_actor_loss = epoch_actor_loss / len(data_loader.dataset)
            epoch_critic_loss = epoch_critic_loss / len(data_loader.dataset)
            self.optimizer_c.zero_grad()
            epoch_critic_loss.backward()
            self.optimizer_c.step()

        return epoch_loss, epoch_actor_loss, epoch_critic_loss, epoch_entropy_loss

    def loss_single_batch(self, old_states, old_actions, old_logprobs, old_state_values, d_rewards, advantages):
        # Calculate surrogate loss
        logprobs, state_values, dist_entropy = self.policy.evaluate(old_states, old_actions)
        state_values = torch.squeeze(state_values, -1)
        ratios = torch.exp(logprobs - old_logprobs.detach())

        surr1 = ratios * advantages
        surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages

        actor_loss = -torch.min(surr1, surr2).sum()
        critic_loss = F.mse_loss(state_values, d_rewards, reduction="sum")
        entropy_loss = -0.005 * dist_entropy.sum()
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
    x2 = img
    img = prev[35:195]
    img = img[::2, ::2, 0]
    img[img == 144] = 0
    img[img == 109] = 0
    img[img != 0] = 1
    x1 = img
    return (x2 - x1).astype(np.float32).ravel()

# Training loop
def train(num_epochs, load_from=None, eval=False):
    render_mode = 'human' if eval else 'rgb_array'
    env = gym.make('Pong-v0', render_mode=render_mode)
    input_shape = env.observation_space.shape
    output_dim = env.action_space.n
    ppo_agent = PPO(input_shape, output_dim, lr_a=LR_A, lr_c=LR_C, gamma=GAMMA, K_epochs=K_EPOCHS, eps_clip=EPS_CLIP, batch_size=BATCH_SIZE)
    ppo_agent.policy.to(device)
    ppo_agent.policy_old.to(device)

    if eval:
        ppo_agent.policy.eval()
        ppo_agent.policy_old.eval()

    writer = SummaryWriter()

    start_epoch = 0
    total_steps = 0

    if load_from is not None:
        saved_dict = torch.load(load_from, map_location=torch.device('cpu') if eval else None)
        ppo_agent.policy.load_state_dict(saved_dict['state_dict'])
        ppo_agent.policy_old.load_state_dict(saved_dict['state_dict'])
        ppo_agent.optimizer_a.load_state_dict(saved_dict['optimizer_a_state_dict'])
        ppo_agent.optimizer_c.load_state_dict(saved_dict['optimizer_c_state_dict'])
        start_epoch = saved_dict['epoch']
        total_steps = saved_dict['total_steps']
        print(f"loaded model from epoch {start_epoch}...")

    for epoch in range(start_epoch, num_epochs):
        curr_frame, _ = env.reset()
        state = preprocess(None, curr_frame)
        total_reward = 0
        step = 0
        ppo_agent.buffer.clear()
        while step < EP_MAX:
            step += 1
            total_steps += 1

            action = ppo_agent.select_action(state, store=not eval)
            next_frame, reward, done, _, _ = env.step(action)
            next_state = preprocess(curr_frame, next_frame)
            if not eval:
                ppo_agent.add_buffer(reward, done)

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
                'optimizer_a_state_dict': ppo_agent.optimizer_a.state_dict(),
                'optimizer_c_state_dict': ppo_agent.optimizer_c.state_dict(),
                'epoch': epoch + 1,
                'total_steps': total_steps,
            }, f'{model_dir}/{model}.{epoch + 1}.pth')

    env.close()

# Run the training loop
if __name__ == "__main__":
    # python ppo_pong.py --eval --model models/ppo_pong.diff.5200.pth
    ap = argparse.ArgumentParser(description='Process args.')
    ap.add_argument('--eval', action='store_true', help='evaluate')
    ap.add_argument('--model', type=str, default=None, help='model to load')
    args = ap.parse_args()

    train(10000, args.model, args.eval)
