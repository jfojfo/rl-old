import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import torchvision
import gym
import argparse


# Hyperparameters
NUM_ENVS = 8
EP_MAX = 20000
EP_LEN = 256
K_EPOCHS = 8
GAMMA = 0.99
EPS_CLIP = 0.2
LR_A = 0.0003
LR_C = 0.0005

model_dir = 'models'
model = 'ppo_pong'

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
    def __init__(self, input_shape, output_dim):
        super(ActorCritic, self).__init__()
        # N = input_shape[0] * input_shape[1] * input_shape[2]
        self.conv_feature = nn.Sequential(
            # 3, 80, 104
            ResidualBlock(3, 16, stride=2),   # 16, 40, 52
            ResidualBlock(16, 32, stride=2),  # 32, 20, 26
            ResidualBlock(32, 32, stride=2),  # 32, 10, 13
            nn.Flatten(),
        )
        N = 32 * 10 * 13
        self.actor = nn.Sequential(
            nn.Linear(N, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, output_dim),
            nn.Softmax(dim=-1),
        )
        self.critic = nn.Sequential(
            nn.Linear(N, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 1),
        )

    # def forward(self, x):
        # action_probs = torch.softmax(self.actor(x), dim=-1)
        # action_probs = self.actor(x)
        # state_values = self.critic(x)
        # return action_probs, state_values

    def act(self, state):
        feature = self.conv_feature(state)
        action_probs = self.actor(feature)
        dist = torch.distributions.Categorical(action_probs)
        action = dist.sample()
        action_logprob = dist.log_prob(action)
        state_val = self.critic(feature).squeeze(-1)
        return action.detach(), action_logprob.detach(), state_val.detach()

    def evaluate(self, states, actions):
        feature = self.conv_feature(states)
        action_probs = self.actor(feature)
        dist = torch.distributions.Categorical(action_probs)
        action_logprobs = dist.log_prob(actions)
        dist_entropy = dist.entropy()
        state_values = self.critic(feature)
        return action_logprobs, state_values, dist_entropy

class RolloutBuffer:
    def __init__(self):
        self.actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []
        self.state_values = []
        self.dones = []

    def clear(self):
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.state_values[:]
        del self.dones[:]


# Define the PPO algorithm
class PPO:
    def __init__(self, input_dim, output_dim, lr_a=0.0003, lr_c=0.001, gamma=0.99, K_epochs=4, eps_clip=0.2):
        self.gamma = gamma
        self.K_epochs = K_epochs
        self.eps_clip = eps_clip
        self.buffer = RolloutBuffer()

        self.policy = ActorCritic(input_dim, output_dim)
        # self.optimizer = optim.Adam([
        #     {'params': self.policy.actor.parameters(), 'lr': lr_a},
        #     {'params': self.policy.critic.parameters(), 'lr': lr_c},
        # ])
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr_c)
        self.policy_old = ActorCritic(input_dim, output_dim)
        self.policy_old.load_state_dict(self.policy.state_dict())  # Copy initial weights

    def select_action(self, state, store=True):
        with torch.no_grad():
            state = torch.FloatTensor(state).unsqueeze(0).to(device)
            action, logprob, state_value = self.policy_old.act(state)
        if store:
            self.buffer.states.append(state.cpu().squeeze(0))
            self.buffer.actions.append(action.cpu().squeeze(0))
            self.buffer.logprobs.append(logprob.cpu().squeeze(0))
            self.buffer.state_values.append(state_value.cpu().squeeze(0))
        return action.cpu().item()

    def add_buffer(self, reward, done):
        self.buffer.rewards.append(reward)
        self.buffer.dones.append(done)

    def compute_value(self, state):
        with torch.no_grad():
            state = torch.FloatTensor(state).unsqueeze(0).to(device)
            action, logprob, state_value = self.policy_old.act(state)
        return state_value.cpu().item()

    def discounted_rewards(self, last_value):
        returns = []
        running_returns = last_value
        for t in reversed(range(len(self.buffer.rewards))):
            if self.buffer.rewards[t] in (-1, 1):
                running_returns = 0
            running_returns = self.buffer.rewards[t] + self.gamma * running_returns * (1.0 - self.buffer.dones[t])
            returns.append(running_returns)
        returns.reverse()
        return returns

    def update(self, d_rewards):
        self.policy_old.load_state_dict(self.policy.state_dict())

        old_states = torch.stack(self.buffer.states, dim=0).detach().to(device)
        old_actions = torch.stack(self.buffer.actions, dim=0).detach().to(device)
        old_logprobs = torch.stack(self.buffer.logprobs, dim=0).detach().to(device)
        old_state_values = torch.stack(self.buffer.state_values, dim=0).detach().to(device)

        # Calculate advantage
        d_rewards = torch.tensor(d_rewards, dtype=torch.float32).to(device)
        # d_rewards = (d_rewards - d_rewards.mean()) / (d_rewards.std() + 1e-7)
        advantages = d_rewards.detach() - old_state_values.detach()

        # PPO policy updates
        for _ in range(self.K_epochs):
            # Calculate surrogate loss
            logprobs, state_values, dist_entropy = self.policy.evaluate(old_states, old_actions)
            state_values = torch.squeeze(state_values)
            ratios = torch.exp(logprobs - old_logprobs.detach())

            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages

            actor_loss = -torch.min(surr1, surr2).mean()
            critic_loss = F.mse_loss(state_values, d_rewards)
            loss = actor_loss + critic_loss

            # Optimize the model
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            return loss, actor_loss, critic_loss

def preprocess(image):
    img = image[1:-1, :, :]
    scale = torchvision.transforms.ToTensor()
    img = scale(img).transpose(1, 2)
    img = torchvision.transforms.functional.resize(img, (int(img.shape[1] / 2), int(img.shape[2] / 2)))
    return img

# Training loop
def train(num_epochs, load_from=None, eval=False):
    render_mode = 'human' if eval else 'rgb_array'
    env = gym.make('Pong-v0', render_mode=render_mode)
    input_shape = env.observation_space.shape
    output_dim = env.action_space.n
    ppo_agent = PPO(input_shape, output_dim, lr_a=LR_A, lr_c=LR_C, gamma=GAMMA, K_epochs=K_EPOCHS, eps_clip=EPS_CLIP)
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
        ppo_agent.optimizer.load_state_dict(saved_dict['optimizer_state_dict'])
        start_epoch = saved_dict['epoch']
        total_steps = saved_dict['total_steps']
        print(f"loaded model from epoch {start_epoch}...")

    for epoch in range(start_epoch, num_epochs):
        state, _ = env.reset()
        state = preprocess(state)
        total_reward = 0
        step = 0
        while step < EP_MAX:
            step += 1
            total_steps += 1

            action = ppo_agent.select_action(state, store=not eval)
            next_state, reward, done, _, _ = env.step(action)
            next_state = preprocess(next_state)
            if not eval:
                ppo_agent.add_buffer(reward + 0.001 if reward == 0 else reward, done)

            state = next_state
            total_reward += reward

            # if step % EP_LEN == 0 or step == EP_MAX or done:
            if done and not eval:
                if len(ppo_agent.buffer.rewards) > 0:
                    # last_value = ppo_agent.compute_value(state)
                    last_value = 0.0
                    d_rewards = ppo_agent.discounted_rewards(last_value)
                    loss, actor_loss, critic_loss = ppo_agent.update(d_rewards)
                    ppo_agent.buffer.clear()

                    writer.add_scalar('Loss/Total Loss', loss.item(), total_steps)
                    writer.add_scalar('Loss/Actor Loss', actor_loss.item(), total_steps)
                    writer.add_scalar('Loss/Critic Loss', critic_loss.item(), total_steps)

            if done:
                break

        print(f"Epoch {epoch+1}, steps: {step}, Total Reward: {total_reward}")

        if (epoch + 1) % 100 == 0 and not eval:
            torch.save({
                'state_dict': ppo_agent.policy.state_dict(),
                'optimizer_state_dict': ppo_agent.optimizer.state_dict(),
                'epoch': epoch + 1,
                'total_steps': total_steps,
            }, f'{model_dir}/{model}.{epoch + 1}.pth')

    env.close()

# Run the training loop
if __name__ == "__main__":
    ap = argparse.ArgumentParser(description='Process args.')
    ap.add_argument('--eval', action='store_true', help='evaluate')
    ap.add_argument('--model', type=str, default=None, help='model to load')
    args = ap.parse_args()

    train(10000, args.model, args.eval)
