import os
import PIL
import gymnasium as gym
import torch
import base64
import imageio
import numpy as np
from pathlib import Path
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from PIL import Image
from torch.distributions import Categorical
from stable_baselines3.common.vec_env import VecVideoRecorder, SubprocVecEnv


#
# https://github.com/rossettisimone/PPO_PONG_DISCRETE/blob/master/PPO_PONG.ipynb
#


ENV_ID = "Pong-v0"
H_SIZE = 256 # hidden size, linear units of the output layer
L_RATE = 1e-5 # learning rate, gradient coefficient for CNN weight update
G_GAE = 0.99 # gamma param for GAE
L_GAE = 0.95 # lambda param for GAE
E_CLIP = 0.2 # clipping coefficient
C_1 = 0.5 # squared loss coefficient
C_2 = 0.01 # entropy coefficient
N = 8 # simultaneous processing environments
T = 256 # PPO steps
M = 64 # mini batch size
K = 10 # PPO epochs
T_EPOCHS = 10 # each T_EPOCH
N_TESTS = 10 # do N_TESTS tests
TARGET_REWARD = 20
TRANSFER_LEARNING = False
PATH='models/Pong-v0_+17.900_5660.dat'


class CNN(nn.Module):
    def __init__(self, num_inputs, num_outputs, hidden_size):
        super(CNN, self).__init__()
        self.critic = nn.Sequential(  # The “Critic” estimates the value function
            nn.Conv2d(in_channels=num_inputs,
                      out_channels=16, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(in_channels=16,
                      out_channels=32, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(in_features=2592, out_features=hidden_size),
            nn.ReLU(),
            nn.Linear(in_features=hidden_size, out_features=1),
        )
        self.actor = nn.Sequential(  # The “Actor” updates the policy distribution in the direction suggested by the Critic (such as with policy gradients)
            nn.Conv2d(in_channels=num_inputs,
                      out_channels=16, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(in_channels=16,
                      out_channels=32, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(in_features=2592, out_features=hidden_size),
            nn.ReLU(),
            nn.Linear(in_features=hidden_size, out_features=num_outputs),
            nn.Softmax(dim=1),
        )

    def forward(self, x):
        value = self.critic(x)
        probs = self.actor(x)
        dist = Categorical(probs)
        return dist, value


def normalize(x):
    x -= x.mean()
    x /= (x.std() + 1e-8) # prevent 0 fraction
    return x


def grey_crop_resize_batch(state):  # deal with batch observations
    states = []
    for i in state:
        img = Image.fromarray(i)
        grey_img = img.convert(mode='L')
        left = 0
        top = 34  # empirically chosen
        right = 160
        bottom = 194  # empiricallly chosen
        cropped_img = grey_img.crop((left, top, right, bottom)) # cropped image of above dimension
        resized_img = cropped_img.resize((84, 84))
        array_2d = np.asarray(resized_img)
        array_3d = np.expand_dims(array_2d, axis=0)
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
    return array_3d # C*H*W


def compute_gae(next_value, rewards, masks, values, gamma=G_GAE, lam=L_GAE):
    values = values + [next_value]  # concat last value to the list
    gae = 0  # first gae always to 0
    returns = []

    for step in reversed(range(len(rewards))):  # for each positions with respect to the result of the action
        delta = rewards[step] + gamma * values[step + 1] * masks[step] - values[
            step]  # compute delta, sum of current reward and the expected goodness of the next state (next state val minus current state val), zero if 'done' is reached, so i can't consider next val
        gae = delta + gamma * lam * masks[
            step] * gae  # recursively compute the sum of the gae until last state is reached, gae is computed summing all gae of previous actions, higher is multiple good actions succeds, lower otherwhise
        returns.insert(0, gae + values[
            step])  # sum again the value of current action, so a state is better to state in if next increment as well
    return returns


def ppo_iter(states, actions, log_probs, returns, advantage):
    batch_size = states.size(0)  # lenght of data collected

    for _ in range(batch_size // M):
        rand_ids = np.random.randint(0, batch_size, M)  # integer array of random indices for selecting M mini batches
        yield states[rand_ids, :], actions[rand_ids, :], log_probs[rand_ids, :], returns[rand_ids, :], advantage[rand_ids, :]


def ppo_update(states, actions, log_probs, returns, advantages, clip_param=E_CLIP):
    for _ in range(K):
        for state, action, old_log_probs, return_, advantage in ppo_iter(states, actions, log_probs, returns, advantages):
            dist, value = model(state)
            action = action.reshape(1, len(action)) # take the relative action and take the column
            new_log_probs = dist.log_prob(action)
            new_log_probs = new_log_probs.reshape(len(old_log_probs), 1) # take the column
            ratio = (new_log_probs - old_log_probs).exp() # new_prob/old_prob
            surr1 = ratio * advantage
            surr2 = torch.clamp(ratio, 1.0 - clip_param, 1.0 + clip_param) * advantage
            actor_loss = - torch.min(surr1, surr2).mean()
            critic_loss = (return_ - value).pow(2).mean()
            entropy = dist.entropy().mean()
            loss = C_1 * critic_loss + actor_loss - C_2 * entropy # loss function clip+vs+f
            optimizer.zero_grad() # in PyTorch, we need to set the gradients to zero before starting to do backpropragation because PyTorch accumulates the gradients on subsequent backward passes.
            loss.backward() # computes dloss/dx for every parameter x which has requires_grad=True. These are accumulated into x.grad for every parameter x
            optimizer.step() # performs the parameters update based on the current gradient and the update rule


def test_env(env, model, device):
    state, _ = env.reset()
    state = grey_crop_resize(state)

    done = False
    total_reward = 0
    while not done:
        state = torch.FloatTensor(state).unsqueeze(0).to(device)
        dist, _ = model(state)
        action = dist.sample().cpu().numpy()[0]
        next_state, reward, done, _, _ = env.step(action)
        next_state = grey_crop_resize(next_state)
        state = next_state
        total_reward += reward
    return total_reward

def ppo_train(model, envs, device, use_cuda, test_rewards, test_epochs, train_epoch, best_reward, early_stop=False):
    env = gym.make('Pong-v0', render_mode='rgb_array')
    state = envs.reset()
    state = grey_crop_resize_batch(state)

    while not early_stop:
        log_probs = []
        values = []
        states = []
        actions = []
        rewards = []
        masks = []

        for _ in range(T):
            state = torch.FloatTensor(state).to(device)
            dist, value = model(state)
            action = dist.sample().cuda() if use_cuda else dist.sample()
            next_state, reward, done, _ = envs.step(action.cpu().numpy())
            next_state = grey_crop_resize_batch(
                next_state)  # simplify perceptions (grayscale-> crop-> resize) to train CNN
            log_prob = dist.log_prob(
                action)  # needed to compute probability ratio r(theta) that prevent policy to vary too much probability related to each action (make the computations more robust)
            log_prob_vect = log_prob.reshape(len(log_prob), 1)  # transpose from row to column
            log_probs.append(log_prob_vect)
            action_vect = action.reshape(len(action), 1)  # transpose from row to column
            actions.append(action_vect)
            values.append(value)
            rewards.append(torch.FloatTensor(reward).unsqueeze(1).to(device))
            masks.append(torch.FloatTensor(1 - done).unsqueeze(1).to(device))
            states.append(state)
            state = next_state

        next_state = torch.FloatTensor(next_state).to(device)  # consider last state of the collection step
        _, next_value = model(next_state)  # collect last value effect of the last collection step
        returns = compute_gae(next_value, rewards, masks, values)
        returns = torch.cat(
            returns).detach()  # concatenates along existing dimension and detach the tensor from the network graph, making the tensor no gradient
        log_probs = torch.cat(log_probs).detach()
        values = torch.cat(values).detach()
        states = torch.cat(states)
        actions = torch.cat(actions)
        advantage = returns - values  # compute advantage for each action
        advantage = normalize(advantage)  # compute the normalization of the vector to make uniform values
        ppo_update(states, actions, log_probs, returns, advantage)
        train_epoch += 1

        if train_epoch % T_EPOCHS == 0:  # do a test every T_EPOCHS times
            test_reward = np.mean(
                [test_env(env, model, device) for _ in range(N_TESTS)])  # do N_TESTS tests and takes the mean reward
            test_rewards.append(test_reward)  # collect the mean rewards for saving performance metric
            test_epochs.append(train_epoch)
            print('Epoch: %s -> Reward: %s' % (train_epoch, test_reward))

            if best_reward is None or best_reward < test_reward:  # save a checkpoint every time it achieves a better reward
                if best_reward is not None:
                    print("Best reward updated: %.3f -> %.3f" % (best_reward, test_reward))
                    name = "%s_%+.3f_%d.dat" % (ENV_ID, test_reward, train_epoch)
                    fname = os.path.join('.', 'PPO_PONG/checkpoints', name)
                    states = {
                        'state_dict': model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'test_rewards': test_rewards,
                        'test_epochs': test_epochs,
                    }
                    torch.save(states,
                               fname)  # save the model, for transfer learning is important to save: model parameters, optimizer parameters, epochs and rewards record as well

                best_reward = test_reward

            if test_reward > TARGET_REWARD:  # stop training if archive the best
                early_stop = True


if __name__ == "__main__":
    for func in [
        lambda: os.mkdir(os.path.join('.', 'PPO_PONG')),
        lambda: os.mkdir(os.path.join('.', 'PPO_PONG/checkpoints')),
        lambda: os.mkdir(os.path.join('.', 'PPO_PONG/records')),
        lambda: os.mkdir(os.path.join('.', 'PPO_PONG/plots'))
    ]:  # create directories
        try:
            func()
        except Exception as error:
            print(error)
            continue

    use_cuda = torch.cuda.is_available()  # Autodetect CUDA
    device = torch.device("cuda" if use_cuda else "cpu")
    print('Device:', device)

    envs = [lambda: gym.make('Pong-v0', render_mode='rgb_array')] * N  # Prepare N actors in N environments
    envs = SubprocVecEnv(
        envs)  # Vectorized Environments are a method for stacking multiple independent environments into a single environment. Instead of the training an RL agent on 1 environment per step, it allows us to train it on n environments per step. Because of this, actions passed to the environment are now a vector (of dimension n). It is the same for observations, rewards and end of episode signals (dones). In the case of non-array observation spaces such as Dict or Tuple, where different sub-spaces may have different shapes, the sub-observations are vectors (of dimension n).
    num_inputs = 1
    num_outputs = envs.action_space.n
    model = CNN(num_inputs, num_outputs, H_SIZE).to(device)
    optimizer = optim.Adam(model.parameters(), lr=L_RATE)  # implements Adam algorithm
    test_rewards = []
    test_epochs = []
    train_epoch = 0
    best_reward = None

    if TRANSFER_LEARNING:  # transfer learning set this variable to True
        if use_cuda:
            checkpoint = torch.load(PATH)
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            test_rewards = checkpoint['test_rewards']
            test_epochs = checkpoint['test_epochs']
            train_epoch = test_epochs[-1]
            best_reward = test_rewards[-1]
        else:
            checkpoint = torch.load(PATH, map_location=lambda storage, loc: storage)
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            test_rewards = checkpoint['test_rewards']
            test_epochs = checkpoint['test_epochs']
            train_epoch = test_epochs[-1]
            best_reward = test_rewards[-1]
        print('CNN: loaded')
        print('Previous best reward: %.3f' % (best_reward))

    print(model)
    print(optimizer)

    ppo_train(model, envs, device, use_cuda, test_rewards, test_epochs, train_epoch, best_reward)
