import torch
import torch.nn as nn
import torch.optim as optim
import gym

# Hyperparameters
epochs = 1000
max_steps = 200
num_envs = 8
gamma = 0.99
eps_clip = 0.2
lr = 0.001
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Create the PingPong environment
envs = [gym.make("Pong-v0") for _ in range(num_envs)]

# Define the policy network
class Policy(nn.Module):
    def __init__(self):
        super(Policy, self).__init__()
        self.fc1 = nn.Linear(80 * 80, 256)
        self.fc2 = nn.Linear(256, 2)
        self.fc3 = nn.Linear(256, 1)  # Value output
        self.relu = nn.ReLU()

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        logits = self.fc2(x)
        value = self.fc3(x)
        return logits, value

# Create the policy network
policy = Policy().to(device)
optimizer = optim.Adam(policy.parameters(), lr=lr)

def compute_returns(rewards, masks, values):
    returns = torch.zeros_like(rewards)
    running_returns = values[-1]
    for t in reversed(range(len(rewards))):
        running_returns = rewards[t] + gamma * running_returns * masks[t]
        returns[t] = running_returns
    return returns

def update_policy(policy, optimizer, observations, actions, returns, advantages):
    for _ in range(epochs):
        # Compute old log probabilities and values
        logits, values = policy(observations)
        dist = torch.distributions.Categorical(logits=logits)
        old_probs = dist.log_prob(actions)
        old_values = values

        # Update policy network using PPO
        for _ in range(max_steps):
            logits, values = policy(observations)
            dist = torch.distributions.Categorical(logits=logits)
            probs = dist.log_prob(actions)
            ratio = (probs - old_probs).exp()

            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - eps_clip, 1 + eps_clip) * advantages
            actor_loss = -torch.min(surr1, surr2).mean()

            critic_loss = nn.MSELoss()(values, returns)

            optimizer.zero_grad()
            loss = actor_loss + critic_loss
            loss.backward()
            optimizer.step()

def train():
    for epoch in range(epochs):
        observations = torch.zeros(num_envs, 80 * 80).to(device)
        actions = torch.zeros(num_envs, 1).long().to(device)
        rewards = torch.zeros(num_envs, 1).to(device)
        masks = torch.zeros(num_envs, 1).to(device)
        values = torch.zeros(num_envs, 1).to(device)
        advantages = torch.zeros(num_envs, 1).to(device)

        # Collect data and compute advantages
        for step in range(max_steps):
            for i, env in enumerate(envs):
                observations[i] = torch.from_numpy(env.render(mode="rgb_array")).float().to(device)
            logits, values = policy(observations)

            dist = torch.distributions.Categorical(logits=logits)
            actions = dist.sample()
            log_probs = dist.log_prob(actions)

            for i, env in enumerate(envs):
                _, rewards[i], done, _ = env.step(actions[i].item())
                masks[i] = 0 if done else 1

            with torch.no_grad():
                next_values = policy(observations)[-1].detach()

            returns = compute_returns(rewards, masks, next_values)
            advantages = returns - values

            update_policy(policy, optimizer, observations, actions, returns, advantages)

# Start training
train()
