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
import torchvision
import plot_util
from collections import OrderedDict


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
C_3 = 1 # image reconstruction loss coefficient
C_4 = 0.01 # kl loss coefficient
N = 2 # simultaneous processing environments
T = 256 # PPO steps to get envs data
M = 64 # mini batch size
K = 10 # PPO epochs repeated to optimise
T_EPOCHS = 100 # T_EPOCH to test and save
N_TESTS = 10 # do N_TESTS tests
TARGET_REWARD = 20
TRANSFER_LEARNING = False
LATENT_DIM = 10

MODEL_DIR = 'models'
MODEL = 'ppo_demo.vae_ac_from_latent_10_c3_1_c4_0.01_recon_sum_mean_kl_sum_mean_no_batchnorm'
# ENV_ID = 'Pong-v0'
ENV_ID = 'PongDeterministic-v0'

class CNN(nn.Module):
    def __init__(self, num_inputs, num_outputs, hidden_size):
        super(CNN, self).__init__()
        self.feature = nn.Sequential(OrderedDict([
            ('conv1', nn.Conv2d(in_channels=num_inputs, out_channels=16, kernel_size=8, stride=4)),
            # ('batchnorm1', nn.BatchNorm2d(16)),
            ('act1', nn.ReLU()),
            ('conv2', nn.Conv2d(in_channels=16, out_channels=32, kernel_size=4, stride=2)),
            # ('batchnorm2', nn.BatchNorm2d(32)),
            ('act2', nn.ReLU()),
            ('flatten', nn.Flatten()),
            ('linear', nn.Linear(in_features=2592, out_features=hidden_size)),
            ('act3', nn.ReLU()),
        ]))
        self.encoder = nn.Sequential(
            nn.Linear(hidden_size, LATENT_DIM * 2)
        )
        self.decoder = nn.Sequential(
            nn.Linear(LATENT_DIM, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 2592),
            nn.ReLU(),
            nn.Unflatten(1, (32, 9, 9)),
            nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2),
            # nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 1, kernel_size=8, stride=4),
            nn.Sigmoid()
        )
        self.critic = nn.Sequential(  # The “Critic” estimates the value function
            nn.Linear(in_features=LATENT_DIM * 2, out_features=hidden_size),
            nn.ReLU(),
            nn.Linear(in_features=hidden_size, out_features=1),
        )
        self.actor = nn.Sequential(  # The “Actor” updates the policy distribution in the direction suggested by the Critic (such as with policy gradients)
            nn.Linear(in_features=LATENT_DIM * 2, out_features=hidden_size),
            nn.ReLU(),
            nn.Linear(in_features=hidden_size, out_features=num_outputs),
            nn.Softmax(dim=1),
        )

    def forward(self, x):
        feature = self.feature(x)

        latent = self.encoder(feature)
        latent_mu, latent_logvar = torch.chunk(latent, 2, dim=1)
        z = self.latent_sample(latent_mu, latent_logvar)
        x_recon = self.decoder(z)

        value = self.critic(latent)
        probs = self.actor(latent)
        dist = Categorical(probs)

        return dist, value, x_recon, latent_mu, latent_logvar

    def latent_sample(self, mu, logvar):
        if self.training:
            # the reparameterization trick
            std = logvar.mul(0.5).exp_()
            eps = torch.empty_like(std).normal_()
            return mu + eps * std
        else:
            return mu


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


def ppo_iter(states, actions, log_probs, returns, advantage):
    batch_size = states.size(0)  # lenght of data collected

    for _ in range(batch_size // M):
        rand_ids = np.random.randint(0, batch_size, M)  # integer array of random indices for selecting M mini batches
        yield states[rand_ids, :], actions[rand_ids, :], log_probs[rand_ids, :], returns[rand_ids, :], advantage[rand_ids, :]


def ppo_update(model, optimizer, states, actions, log_probs, returns, advantages, clip_param=E_CLIP):
    params_feature = model.get_parameter('feature.linear.weight')
    for _ in range(K):
        for state, action, old_log_probs, return_, advantage in ppo_iter(states, actions, log_probs, returns, advantages):
            dist, value, state_recon, latent_mu, latent_logvar = model(state)
            action = action.reshape(1, len(action)) # take the relative action and take the column
            new_log_probs = dist.log_prob(action)
            new_log_probs = new_log_probs.reshape(len(old_log_probs), 1) # take the column
            ratio = (new_log_probs - old_log_probs).exp() # new_prob/old_prob
            surr1 = ratio * advantage
            surr2 = torch.clamp(ratio, 1.0 - clip_param, 1.0 + clip_param) * advantage
            actor_loss = - torch.min(surr1, surr2).mean()
            critic_loss = (return_ - value).pow(2).mean()
            entropy_loss = dist.entropy().mean()
            recon_loss = F.binary_cross_entropy(state_recon, state, reduction='none').sum(dim=(1,2,3)).mean()
            # recon_loss = (state_recon - state).pow(2).sum(dim=(1, 2, 3)).mean()
            kl_loss = -(1 + latent_logvar - latent_mu.pow(2) - latent_logvar.exp()).sum(dim=1).mean()
            loss = C_1 * critic_loss + actor_loss - C_2 * entropy_loss + C_3 * recon_loss + C_4 * kl_loss # loss function clip+vs+f

            # np_latent_mu = latent_mu.clone().detach().numpy()
            # np_latent_var = latent_logvar.clone().exp().detach().numpy()

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
            (C_3 * recon_loss).backward(retain_graph=True)
            grad_recon = params_feature.grad.mean(), params_feature.grad.std()

            optimizer.zero_grad()
            (C_4 * kl_loss).backward(retain_graph=True)
            grad_kl = params_feature.grad.mean(), params_feature.grad.std()

            optimizer.zero_grad()
            loss.backward()
            # max_norm = 100.0
            # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)

            grad_total = params_feature.grad.mean(), params_feature.grad.std()
            grad_max = params_feature.grad.abs().max()

            # loss.backward() # computes dloss/dx for every parameter x which has requires_grad=True. These are accumulated into x.grad for every parameter x
            optimizer.step() # performs the parameters update based on the current gradient and the update rule
    return loss, actor_loss, critic_loss, entropy_loss, recon_loss / (state.shape[2] * state.shape[3]), kl_loss / LATENT_DIM, \
           grad_critic[0], grad_actor[0], grad_entropy[0], grad_recon[0], grad_kl[0], grad_total[0], grad_max, \
           latent_mu, latent_logvar.exp(), state, state_recon


def test_env(env, model, device):
    state, _ = env.reset()
    state = grey_crop_resize(state)

    done = False
    total_reward = 0
    while not done:
        state = torch.FloatTensor(state).unsqueeze(0).to(device)
        dist, _, _, _, _ = model(state)
        action = dist.sample().cpu().numpy()[0]
        next_state, reward, done, _, _ = env.step(action)
        next_state = grey_crop_resize(next_state)
        state = next_state
        total_reward += reward
    return total_reward

def hook_fn(module, input, output):
    module.feature_map = output

def ppo_train(model, envs, device, optimizer, test_rewards, test_epochs, train_epoch, best_reward, early_stop=False):
    conv1_model = model.feature.get_submodule('conv1')
    conv1_model.register_forward_hook(hook_fn)
    conv2_model = model.feature.get_submodule('conv2')
    conv2_model.register_forward_hook(hook_fn)

    writer = SummaryWriter(comment=f'.{MODEL}.{ENV_ID}')
    env_test = gym.make(ENV_ID, render_mode='rgb_array')

    state = envs.reset()
    state = grey_crop_resize_batch(state)

    total_reward_1_env = 0
    total_runs_1_env = 0
    steps_1_env = 0
    last_save_epoch = 0

    while not early_stop:
        log_probs = []
        values = []
        states = []
        actions = []
        rewards = []
        masks = []

        model.eval()
        for i in range(T):
            state = torch.FloatTensor(state).to(device)
            dist, value, _, _, _ = model(state)
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

            total_reward_1_env += reward[0]
            steps_1_env += 1
            if done[0]:
                total_runs_1_env += 1
                print(f'Run {total_runs_1_env}, steps {steps_1_env}, Reward {total_reward_1_env}')
                writer.add_scalar('Reward/train_reward_1_env', total_reward_1_env, train_epoch * T + i + 1)
                total_reward_1_env = 0
                steps_1_env = 0

        next_state = torch.FloatTensor(next_state).to(device)  # consider last state of the collection step
        _, next_value, _, _, _ = model(next_state)  # collect last value effect of the last collection step
        returns = compute_gae(next_value, rewards, masks, values)
        returns = torch.cat(returns).detach()  # concatenates along existing dimension and detach the tensor from the network graph, making the tensor no gradient
        log_probs = torch.cat(log_probs).detach()
        values = torch.cat(values).detach()
        states = torch.cat(states)
        actions = torch.cat(actions)
        advantage = returns - values  # compute advantage for each action
        advantage = normalize(advantage)  # compute the normalization of the vector to make uniform values

        model.train()
        loss, actor_loss, critic_loss, entropy_loss, recon_loss, kl_loss, \
        grad_critic_mean, grad_actor_mean, grad_entropy_mean, grad_recon_mean, grad_kl_mean, grad_total_mean, grad_max,\
        latent_mu, latent_var, x, x_recon = \
            ppo_update(model, optimizer, states, actions, log_probs, returns, advantage)
        model.eval()

        train_epoch += 1

        total_steps = train_epoch * T
        writer.add_scalar('Loss/Total Loss', loss.item(), total_steps)
        writer.add_scalar('Loss/Actor Loss', actor_loss.item(), total_steps)
        writer.add_scalar('Loss/Critic Loss', critic_loss.item(), total_steps)
        writer.add_scalar('Loss/Entropy Loss', entropy_loss.item(), total_steps)
        writer.add_scalar('Loss/Recon Loss', recon_loss.item(), total_steps)
        writer.add_scalar('Loss/KL Loss', kl_loss.item(), total_steps)

        writer.add_scalar('Grad/Max', grad_max.item(), total_steps)
        writer.add_scalar('Grad/Critic', grad_critic_mean.item(), total_steps)
        writer.add_scalar('Grad/Actor', grad_actor_mean.item(), total_steps)
        writer.add_scalar('Grad/Entropy', grad_entropy_mean.item(), total_steps)
        writer.add_scalar('Grad/Recon', grad_recon_mean.item(), total_steps)
        writer.add_scalar('Grad/KL', grad_kl_mean.item(), total_steps)
        writer.add_scalar('Grad/Total', grad_total_mean.item(), total_steps)

        for i, mu in enumerate(latent_mu[0]):
            writer.add_scalar(f'Latent/mu_{i}', mu.item(), total_steps)
        for i, var in enumerate(latent_var[0]):
            writer.add_scalar(f'Latent/var_{i}', var.item(), total_steps)

        if train_epoch % T_EPOCHS == 0:  # do a test every T_EPOCHS times
            # 增维，（batch_num,output_channel,width,height）->(batch_num,output_channel,1,width,height)
            # (64, 16, 20, 20)
            b, c, w, h = conv1_model.feature_map.shape
            conv1_feature_map = conv1_model.feature_map[:16].unsqueeze(dim=2).view(-1, 1, w, h)
            conv1_feature_map_grids = torchvision.utils.make_grid(conv1_feature_map, nrow=16, padding=1, normalize=True)
            writer.add_image("FeatureMap/conv1", conv1_feature_map_grids, total_steps)

            # (64, 32, 9, 9)
            b, c, w, h = conv2_model.feature_map.shape
            conv2_feature_map = conv2_model.feature_map.unsqueeze(dim=2).view(-1, 1, w, h)
            conv2_feature_map_grids = torchvision.utils.make_grid(conv2_feature_map, nrow=32, padding=1, normalize=True)
            writer.add_image("FeatureMap/conv2", conv2_feature_map_grids, total_steps)

            for i in range(4):
                writer.add_images(f'Image/{i}', torch.stack((x[i], x_recon[i]), dim=0), total_steps)

            test_reward = np.mean([test_env(env_test, model, device) for _ in range(N_TESTS)])  # do N_TESTS tests and takes the mean reward
            test_rewards.append(test_reward)  # collect the mean rewards for saving performance metric
            test_epochs.append(train_epoch)
            print('Epoch: %s -> Reward: %s' % (train_epoch, test_reward))
            writer.add_scalar('Reward/test_reward', test_reward, total_steps)

            for i in range(0, LATENT_DIM - 1, 2):
                latent_recon = plot_util.plot_latent_space(model, LATENT_DIM, i, (-2, 2), 15)
                writer.add_image(f"Image/latent_recon_{i}", latent_recon, total_steps)

            if best_reward is None or best_reward < test_reward:  # save a checkpoint every time it achieves a better reward
                if best_reward is not None:
                    print("Best reward updated: %.3f -> %.3f" % (best_reward, test_reward))
                    save_model(model, optimizer, test_reward, train_epoch, test_rewards, test_epochs)
                    last_save_epoch = train_epoch
                best_reward = test_reward
            elif train_epoch - last_save_epoch >= 500:
                print("save model: %.3f" % test_reward)
                save_model(model, optimizer, test_reward, train_epoch, test_rewards, test_epochs)
                last_save_epoch = train_epoch

            if test_reward > TARGET_REWARD:  # stop training if archive the best
                early_stop = True


def save_model(model, optimizer, test_reward, train_epoch, test_rewards, test_epochs):
    name = "%s_%s_%+.3f_%d.pth" % (MODEL, ENV_ID, test_reward, train_epoch)
    fname = os.path.join(MODEL_DIR, name)
    states = {
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'test_rewards': test_rewards,
        'test_epochs': test_epochs,
    }
    torch.save(states, fname)


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
    model = CNN(num_inputs, num_outputs, H_SIZE).to(device)
    optimizer = optim.Adam(model.parameters(), lr=L_RATE)  # implements Adam algorithm
    test_rewards = []
    test_epochs = []
    train_epoch = 0
    best_reward = None

    summary(model, input_size=[(1, 84, 84)], batch_dim=0, dtypes=[torch.float])

    if load_from is not None:
        checkpoint = torch.load(load_from, map_location=None if use_cuda else torch.device('cpu'))
        state_dict = checkpoint['state_dict']
        # adapt model layer's old name
        if 'feature.conv1.weight' not in state_dict:
            state_dict['feature.conv1.weight'] = state_dict['feature.0.weight']
            state_dict['feature.conv1.bias'] = state_dict['feature.0.bias']
            del state_dict['feature.0.weight']
            del state_dict['feature.0.bias']
        if 'feature.conv2.weight' not in state_dict:
            state_dict['feature.conv2.weight'] = state_dict['feature.2.weight']
            state_dict['feature.conv2.bias'] = state_dict['feature.2.bias']
            del state_dict['feature.2.weight']
            del state_dict['feature.2.bias']
        if 'feature.linear.weight' not in state_dict:
            state_dict['feature.linear.weight'] = state_dict['feature.5.weight']
            state_dict['feature.linear.bias'] = state_dict['feature.5.bias']
            del state_dict['feature.5.weight']
            del state_dict['feature.5.bias']
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


def eval(load_from):
    use_cuda = torch.cuda.is_available()  # Autodetect CUDA
    device = torch.device("cuda" if use_cuda else "cpu")
    print('Device:', device)

    env_test = gym.make(ENV_ID, render_mode='human')

    num_inputs = 1
    num_outputs = env_test.action_space.n
    model = CNN(num_inputs, num_outputs, H_SIZE).to(device)
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
