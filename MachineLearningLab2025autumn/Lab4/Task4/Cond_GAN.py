import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST
from tqdm.auto import tqdm


torch.manual_seed(0)


def get_one_hot_labels(labels, n_classes):
    """Return one-hot encoded labels."""
    return F.one_hot(labels, n_classes)


def combine_vectors(x, y):
    """Concatenate two tensors along feature dimension and return float tensor."""
    return torch.cat((x.float(), y.float()), dim=1)


def get_generator_block(input_dim, output_dim):
    return nn.Sequential(
        nn.Linear(input_dim, output_dim),
        nn.BatchNorm1d(output_dim),
        nn.ReLU(inplace=True),
    )


def get_discriminator_block(input_dim, output_dim):
    return nn.Sequential(
        nn.Linear(input_dim, output_dim),
        nn.LeakyReLU(0.2, inplace=True),
    )


class Generator(nn.Module):
    def __init__(self, z_dim=64, n_classes=10, im_dim=784, hidden_dim=128):
        super().__init__()
        input_dim = z_dim + n_classes
        self.gen = nn.Sequential(
            get_generator_block(input_dim, hidden_dim),
            get_generator_block(hidden_dim, hidden_dim * 2),
            get_generator_block(hidden_dim * 2, hidden_dim * 4),
            get_generator_block(hidden_dim * 4, hidden_dim * 8),
            nn.Linear(hidden_dim * 8, im_dim),
            nn.Sigmoid(),
        )

    def forward(self, noise_and_labels):
        return self.gen(noise_and_labels)


class Discriminator(nn.Module):
    def __init__(self, n_classes=10, im_dim=784, hidden_dim=128):
        super().__init__()
        input_dim = im_dim + n_classes
        self.disc = nn.Sequential(
            get_discriminator_block(input_dim, hidden_dim * 4),
            get_discriminator_block(hidden_dim * 4, hidden_dim * 2),
            get_discriminator_block(hidden_dim * 2, hidden_dim),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, image_and_labels):
        return self.disc(image_and_labels)


def get_noise(n_samples, z_dim, device):
    return torch.randn(n_samples, z_dim, device=device)


def train_cond_gan(

    n_epochs=50,
    z_dim=64,
    n_classes=10,
    batch_size=128,
    lr=2e-4,
    device=None,
):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    transform = transforms.ToTensor()
    dataloader = DataLoader(
        MNIST(root=".", download=True, transform=transform),
        batch_size=batch_size,
        shuffle=True,
    )

    gen = Generator(z_dim=z_dim, n_classes=n_classes).to(device)
    disc = Discriminator(n_classes=n_classes).to(device)
    gen_opt = torch.optim.Adam(gen.parameters(), lr=lr)
    disc_opt = torch.optim.Adam(disc.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss()

    display_step = 500
    cur_step = 0
    mean_generator_loss = 0.0
    mean_discriminator_loss = 0.0

    for epoch in range(n_epochs):
        for real, labels in tqdm(dataloader, leave=False):
            cur_batch_size = real.size(0)
            real = real.view(cur_batch_size, -1).to(device)
            labels = labels.to(device)
            one_hot = get_one_hot_labels(labels, n_classes).float()

            # Update discriminator
            disc_opt.zero_grad()
            noise = get_noise(cur_batch_size, z_dim, device)
            noise_and_labels = combine_vectors(noise, one_hot)
            fake = gen(noise_and_labels)

            fake_image_and_labels = combine_vectors(fake.detach(), one_hot)
            real_image_and_labels = combine_vectors(real, one_hot)

            fake_pred = disc(fake_image_and_labels)
            real_pred = disc(real_image_and_labels)
            fake_loss = criterion(fake_pred, torch.zeros_like(fake_pred))
            real_loss = criterion(real_pred, torch.ones_like(real_pred))
            disc_loss = (fake_loss + real_loss) / 2
            disc_loss.backward()
            disc_opt.step()

            # Update generator
            gen_opt.zero_grad()
            fake_pred = disc(combine_vectors(fake, one_hot))
            gen_loss = criterion(fake_pred, torch.ones_like(fake_pred))
            gen_loss.backward()
            gen_opt.step()

            mean_generator_loss += gen_loss.item() / display_step
            mean_discriminator_loss += disc_loss.item() / display_step

            if cur_step % display_step == 0 and cur_step > 0:
                print(
                    f"Epoch {epoch} Step {cur_step}: "
                    f"Gen loss {mean_generator_loss:.4f} | Disc loss {mean_discriminator_loss:.4f}"
                )
                mean_generator_loss = 0.0
                mean_discriminator_loss = 0.0
            cur_step += 1

    return gen, disc


if __name__ == "__main__":
    train_cond_gan()
