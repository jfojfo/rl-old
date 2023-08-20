import numpy as np
import torch
import torchvision


def plot_latent_space(model, latent_dim, latent_index=0, range=(-2.5, 2.5), nrow=10):
    device = next(model.parameters()).device

    # create a sample grid in 2d latent space
    latent_x = np.linspace(range[0], range[1], nrow)
    latent_y = np.linspace(range[0], range[1], nrow)
    latents = torch.FloatTensor(len(latent_y), len(latent_x), 2)
    for i, lx in enumerate(latent_x):
        for j, ly in enumerate(latent_y):
            latents[j, i, 0] = lx
            latents[j, i, 1] = ly
    latents = latents.view(-1, 2) # flatten grid into a batch
    latents_all = torch.zeros((latents.shape[0], latent_dim)) + 0.0
    # latents = torch.cat((latents_other, latents), dim=1)
    latents_all[:, latent_index:latent_index+2] = latents

    mixed = latents_all.to(device)
    image_recon = model.decoder(mixed)
    image_recon = image_recon.cpu()

    return torchvision.utils.make_grid(image_recon.data, nrow, 1)
    # fig, ax = plt.subplots(figsize=(42, 42))
    # show_image(torchvision.utils.make_grid(image_recon.data[:100],10,1), ax, "")
    # plt.savefig('sample.pdf', dpi=300, format='pdf')
    # plt.show()
