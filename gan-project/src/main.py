import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import torch
from models import make_generator_network, make_discriminator_network
from train import d_train, g_train
from data.dataset import create_mnist_dataloader
from utils import create_noise

def main():
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Hyperparameters
    z_size = 20
    batch_size = 64
    num_epochs = 100
    mode_z = 'normal'
    loss_fn = torch.nn.BCELoss()

    # Initialize models
    gen_model = make_generator_network(input_size=z_size).to(device)
    disc_model = make_discriminator_network(input_size=784).to(device)

    # Create optimizers
    d_optimizer = torch.optim.Adam(disc_model.parameters(), lr=0.0002)
    g_optimizer = torch.optim.Adam(gen_model.parameters(), lr=0.0002)

    # Create DataLoader
    dataloader = create_mnist_dataloader(batch_size)

    # Training loop
    for epoch in range(num_epochs):
        for real_images, _ in dataloader:
            d_loss, d_proba_real, d_proba_fake = d_train(real_images, disc_model, loss_fn, gen_model, create_noise, device, d_optimizer)
            g_loss = g_train(real_images, gen_model, disc_model, loss_fn, create_noise, device, g_optimizer)

        print(f'Epoch {epoch + 1}/{num_epochs} | D Loss: {d_loss:.4f} | G Loss: {g_loss:.4f}')

if __name__ == "__main__":
    main()