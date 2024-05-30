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


# Training function
def train(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    for images, labels in tqdm(dataloader, desc="Training", leave=False):
        images, labels = images.to(device), labels.to(device)

        labels = labels.view(-1)
        outputs = model(images)
        loss = criterion(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running_loss += loss.cpu().item()
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
            outputs = model(images)
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
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5]),
        ]
    )
    train_dataset = PathMNIST(split="train", transform=transform, download=True, as_rgb=True)
    train_dataset = create_exponentially_imbalanced_dataset(train_dataset, 100)
    val_dataset = PathMNIST(split="val", transform=transform, download=True, as_rgb=True)
    test_dataset = PathMNIST(split="test", transform=transform, download=True, as_rgb=True)

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
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 9)
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Train and evaluate
    num_epochs = 10
    val_accuracies, test_accuracies = [], []
    for epoch in range(num_epochs):
        train_loss = train(model, train_loader, F.cross_entropy, optimizer, device)
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
