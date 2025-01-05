import torch

def d_train(real_images, disc_model, loss_fn, gen_model, create_noise, device, d_optimizer):
    batch_size = real_images.size(0)
    real_images = real_images.to(device)
    d_labels_real = torch.ones(batch_size, 1, device=device)
    d_labels_fake = torch.zeros(batch_size, 1, device=device)

    # Train on real images
    d_optimizer.zero_grad()
    d_output_real = disc_model(real_images.view(batch_size, -1))
    d_loss_real = loss_fn(d_output_real, d_labels_real)
    d_loss_real.backward()

    # Train on fake images
    noise = create_noise(batch_size, gen_model[0].in_features, mode_z='normal').to(device)
    fake_images = gen_model(noise)
    d_output_fake = disc_model(fake_images.detach())
    d_loss_fake = loss_fn(d_output_fake, d_labels_fake)
    d_loss_fake.backward()
    d_optimizer.step()

    d_loss = d_loss_real + d_loss_fake
    d_proba_real = d_output_real.mean().item()
    d_proba_fake = d_output_fake.mean().item()

    return d_loss.item(), d_proba_real, d_proba_fake

def g_train(real_images, gen_model, disc_model, loss_fn, create_noise, device, g_optimizer):
    batch_size = real_images.size(0)
    g_labels = torch.ones(batch_size, 1, device=device)

    g_optimizer.zero_grad()
    noise = create_noise(batch_size, gen_model[0].in_features, mode_z='normal').to(device)
    fake_images = gen_model(noise)
    d_output_fake = disc_model(fake_images)
    g_loss = loss_fn(d_output_fake, g_labels)
    g_loss.backward()
    g_optimizer.step()

    return g_loss.item()