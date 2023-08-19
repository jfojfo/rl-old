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
import matplotlib.pyplot as plt
from ppo_demo_vae import *
import torchvision


def to_img(x):
    x = x.clamp(0, 1)
    return x

def show_image(img, ax, title):
    img = to_img(img)
    npimg = img.numpy()
    ax.set_title(title)
    ax.imshow(np.transpose(npimg, (1, 2, 0)))

def plot_latent_space(model, device):
    # create a sample grid in 2d latent space
    latent_x = np.linspace(-2.5, 2.5, 10)
    latent_y = np.linspace(-2.5, 2.5, 10)
    latents = torch.FloatTensor(len(latent_y), len(latent_x), 2)
    for i, lx in enumerate(latent_x):
        for j, ly in enumerate(latent_y):
            latents[j, i, 0] = lx
            latents[j, i, 1] = ly
    latents = latents.view(-1, 2) # flatten grid into a batch
    latents_other = torch.zeros((latents.shape[0], LATENT_DIM - latents.shape[1])) + 0.0
    # latents = torch.cat((latents_other, latents), dim=1)
    latents = torch.cat((latents_other[:,0:5], latents[:,:1], latents_other[:,5:8], latents[:,1:], latents_other[:,8:]), dim=1)

    mixed = latents.to(device)
    image_recon = model.decoder(mixed)
    image_recon = image_recon.cpu()

    fig, ax = plt.subplots(figsize=(42, 42))
    show_image(torchvision.utils.make_grid(image_recon.data[:100],10,1), ax, "")
    # plt.savefig('sample.pdf', dpi=300, format='pdf')
    plt.show()


if __name__ == "__main__":
    # python ppo_pong.py --eval --model models/ppo_pong.diff.5200.pth
    ap = argparse.ArgumentParser(description='Process args.')
    ap.add_argument('--model', type=str, default=None, help='model to load')
    args = ap.parse_args()

    use_cuda = torch.cuda.is_available()  # Autodetect CUDA
    device = torch.device("cuda" if use_cuda else "cpu")
    print('Device:', device)

    env = gym.make(ENV_ID, render_mode='rgb_array')
    num_inputs = 1
    num_outputs = env.action_space.n
    model = CNN(num_inputs, num_outputs, H_SIZE).to(device)

    print(f"loading {args.model}")
    checkpoint = torch.load(args.model, map_location=None if use_cuda else torch.device('cpu'))
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()

    plot_latent_space(model, device)
