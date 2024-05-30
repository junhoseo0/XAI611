import os
import random
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import torchvision.transforms as transforms
from medmnist import PathMNIST
from torch.nn import functional as F
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm


class JointSSL(nn.Module):
    def __init__(self, projection_dim=128):
        super().__init__()

        # Here, "num_classes" does not correspond to actual # of classes.
        # Instead, it is a embedding dimension of projection layer.
        self.resnet18 = models.resnet18()
        self.feature_dim = self.resnet18.fc.in_features

        # Change shapes to match SimCLR (which used CIFAR-10).
        self.resnet18.conv1 = nn.Conv2d(3, 64, 3, 1, 1, bias=False)
        self.resnet18.maxpool = nn.Identity()
        self.resnet18.fc = nn.Identity()

        self.projector = nn.Sequential(
            nn.Linear(self.feature_dim, 2048),
            nn.ReLU(),
            nn.Linear(2048, projection_dim),
        )
        self.fc = nn.Sequential(nn.Linear(self.feature_dim, 2048), nn.ReLU(), nn.Linear(2048, 9))

    def forward(self, x):
        feature = self.resnet18(x)
        return self.fc(feature), self.projector(feature)


def get_color_distortion(s=0.5):
    color_jitter = transforms.ColorJitter(0.8 * s, 0.8 * s, 0.8 * s, 0.2 * s)
    rand_color_jitter = transforms.RandomApply([color_jitter], p=0.8)
    rand_gray = transforms.RandomGrayscale(p=0.2)
    return transforms.Compose([rand_color_jitter, rand_gray])


# Training function
def train(model, dataloader, criterion, optimizer, device, temperature=0.5, lamb=0.5):
    model.train()

    rand_transform = transforms.Compose(
        [
            transforms.RandomResizedCrop(28),
            transforms.RandomHorizontalFlip(p=0.5),
        ]
    )

    running_loss = 0.0
    for images, labels in tqdm(dataloader, desc="Training", leave=False):
        # 1. Pass two random transformation of the batch to the model.
        images_0, images_1 = rand_transform(images), rand_transform(images)
        images_ssl = torch.cat([images_0, images_1]).to(device)
        _, projection = model(images_ssl)

        # 2. Compute Normalized Temperature-scaled Cross Entropy Loss (NT-Xent).
        projection = F.normalize(projection)
        similarity = torch.clamp(projection @ projection.t(), min=1e-7) / temperature
        similarity -= 1e5 * torch.eye(similarity.size(0), device=device)

        targets = torch.arange(similarity.size(0), dtype=torch.int64, device=device)
        targets[::2] += 1
        targets[1::2] -= 1

        nt_xent_loss = F.cross_entropy(similarity, targets)

        # 3. Compute the normal SL loss.
        logit, _ = model(images.to(device))
        loss = nt_xent_loss + lamb * criterion(logit, labels.view(-1).to(device))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    return running_loss / len(dataloader)


# Testing function
def test(model, dataloader, criterion, device):
    model.eval()
    correct = 0
    total = 0
    running_loss = 0.0
    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc="Testing", leave=False):
            images, labels = images.to(device), labels.to(device)

            labels = labels.view(-1)
            outputs, _ = model(images)
            loss = criterion(outputs, labels)
            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().cpu().item()
    accuracy = 100 * correct / total
    return running_loss / len(dataloader), accuracy


def create_exponentially_imbalanced_dataset(dataset, imbalance_ratio):
    n_classes = 9
    count = 7000
    new_counts = [
        int(count * (imbalance_ratio ** (-i / (n_classes - 1)))) for i in range(n_classes)
    ]
    labels = torch.tensor(dataset.labels).view(-1)
    indices = []

    for i, count in enumerate(new_counts):
        class_indices = np.where(labels.numpy() == i)[0]
        selected_indices = random.sample(class_indices.tolist(), count)
        indices.extend(selected_indices)

    return Subset(dataset, indices)


def seed_worker(worker_id):
    """Seed a worker with the same seed as the main process."""
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def main(args: list[str]):
    if len(args) > 1:
        raise ValueError("Too many arguments")

    # Reproducibility Setup
    # - Seed Python, Numpy, and PyTorch
    seed = int(args[0])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    # - Use deterministic CUDA.
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    torch.use_deterministic_algorithms(True)

    # Load the dataset.
    train_dataset = PathMNIST(
        split="train", transform=transforms.ToTensor(), download=True, as_rgb=True
    )
    train_dataset = create_exponentially_imbalanced_dataset(train_dataset, 100)
    val_dataset = PathMNIST(
        split="val", transform=transforms.ToTensor(), download=True, as_rgb=True
    )
    test_dataset = PathMNIST(
        split="test", transform=transforms.ToTensor(), download=True, as_rgb=True
    )

    # Prepare the dataloader, and seed them for reproducibility.
    g = torch.Generator()
    g.manual_seed(seed)
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=64,
        shuffle=True,
        num_workers=2,
        worker_init_fn=seed_worker,
        generator=g,
    )
    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=64,
        shuffle=False,
        num_workers=2,
        worker_init_fn=seed_worker,
        generator=g,
    )
    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=64,
        shuffle=False,
        num_workers=2,
        worker_init_fn=seed_worker,
        generator=g,
    )

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    # Initialize the model * the optimizer.
    model = JointSSL(projection_dim=128).to(device=device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Train and evaluate
    num_epochs = 10
    val_accuracies, test_accuracies = [], []
    for epoch in range(num_epochs):
        train_loss = train(model, train_loader, F.cross_entropy, optimizer, device, lamb=10.0)
        val_loss, val_accuracy = test(model, val_loader, F.cross_entropy, device)
        test_loss, test_accuracy = test(model, test_loader, F.cross_entropy, device)

        val_accuracies.append(val_accuracy)
        test_accuracies.append(test_accuracy)
        print(
            f"Epoch [{epoch + 1}/{num_epochs}] Train Loss: {train_loss:.4f},",
            f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.2f}%,",
            f"Test Loss: {test_loss: .4f}, Test Accuracy: {test_accuracy: .4f}%",
        )

    # Report the test accuracy with the best validation accuracy.
    print(
        f"Best Validation Accuracy: {max(val_accuracies)}",
        f"with Test Accurcy of {test_accuracies[np.argmax(val_accuracies)]}",
    )


if __name__ == "__main__":
    main(sys.argv[1:])
