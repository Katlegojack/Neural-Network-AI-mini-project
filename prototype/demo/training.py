import gzip
import random
import struct
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from .model_service import MNISTNetwork


SEED = 42
BATCH_SIZE = 128
EPOCHS = 5
LR = 0.001
EWC_LAMBDA = 1000.0
FISHER_BATCHES = 50
REPLAY_BUFFER_SIZE = 1000
REPLAY_BATCH_SIZE = 64

TASKS = {
    1: (0, 1),
    2: (2, 3),
    3: (4, 5),
    4: (6, 7),
    5: (8, 9),
}


def set_seed():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)


def load_images(path: Path):
    with gzip.open(path, "rb") as f:
        magic, n, rows, cols = struct.unpack(">IIII", f.read(16))
        if magic != 2051:
            raise ValueError("Invalid MNIST image file.")
        return (
            np.frombuffer(f.read(), dtype=np.uint8)
            .reshape(n, rows, cols)
            .astype(np.float32)
            / 255.0
        )


def load_labels(path: Path):
    with gzip.open(path, "rb") as f:
        magic, n = struct.unpack(">II", f.read(8))
        if magic != 2049:
            raise ValueError("Invalid MNIST label file.")
        return np.frombuffer(f.read(), dtype=np.uint8)


def load_training_data(repo_root: Path):
    x = load_images(repo_root / "train-images-idx3-ubyte.gz")
    y = load_labels(repo_root / "train-labels-idx1-ubyte.gz")

    task_arrays = {}
    loaders = {}

    for task_id, digits in TASKS.items():
        mask = np.isin(y, digits)
        tx = x[mask]
        ty = y[mask]
        task_arrays[task_id] = (tx, ty)

        dataset = TensorDataset(
            torch.from_numpy(tx).unsqueeze(1),
            torch.from_numpy(ty.astype(np.int64)),
        )
        loaders[task_id] = DataLoader(
            dataset,
            batch_size=BATCH_SIZE,
            shuffle=True,
            num_workers=0,
        )

    return task_arrays, loaders


def train_standard_task(model, loader, optimizer, criterion):
    for _ in range(EPOCHS):
        model.train()
        for images, labels in loader:
            optimizer.zero_grad()
            loss = criterion(model(images), labels)
            loss.backward()
            optimizer.step()


def train_naive(loaders):
    set_seed()
    model = MNISTNetwork()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()

    for task_id in range(1, 6):
        train_standard_task(model, loaders[task_id], optimizer, criterion)

    return model


def calculate_fisher(model, loader):
    criterion = nn.CrossEntropyLoss()
    fisher = {
        name: torch.zeros_like(param)
        for name, param in model.named_parameters()
    }

    batches = 0
    model.eval()

    for batch_index, (images, labels) in enumerate(loader):
        if batch_index >= FISHER_BATCHES:
            break

        model.zero_grad()
        loss = criterion(model(images), labels)
        loss.backward()

        for name, param in model.named_parameters():
            if param.grad is not None:
                fisher[name] += param.grad.detach() ** 2
        batches += 1

    for name in fisher:
        fisher[name] /= batches

    return fisher


def ewc_penalty(model, memories):
    penalty = torch.tensor(0.0)

    for memory in memories:
        for name, param in model.named_parameters():
            penalty += torch.sum(
                memory["fisher"][name]
                * (param - memory["params"][name]) ** 2
            )

    return penalty


def train_ewc(loaders):
    set_seed()
    model = MNISTNetwork()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()
    memories = []

    for task_id in range(1, 6):
        for _ in range(EPOCHS):
            model.train()
            for images, labels in loaders[task_id]:
                optimizer.zero_grad()
                task_loss = criterion(model(images), labels)
                penalty = ewc_penalty(model, memories)
                loss = task_loss + (EWC_LAMBDA / 2.0) * penalty
                loss.backward()
                optimizer.step()

        if task_id < 5:
            fisher = calculate_fisher(model, loaders[task_id])
            memories.append(
                {
                    "fisher": {
                        name: value.detach().clone()
                        for name, value in fisher.items()
                    },
                    "params": {
                        name: param.detach().clone()
                        for name, param in model.named_parameters()
                    },
                }
            )

    return model


class ReplayBuffer:
    def __init__(self):
        self.rng = np.random.default_rng(SEED)
        self.images = torch.empty((0, 28, 28), dtype=torch.float32)
        self.labels = torch.empty((0,), dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def sample(self):
        if len(self) == 0:
            return None, None

        size = min(REPLAY_BATCH_SIZE, len(self))
        indices = self.rng.choice(len(self), size=size, replace=False)
        indices = torch.tensor(indices, dtype=torch.long)
        return self.images[indices].unsqueeze(1), self.labels[indices]

    def update(self, task_data, seen_classes):
        images, labels = task_data

        if len(self):
            all_images = np.concatenate([self.images.numpy(), images])
            all_labels = np.concatenate([self.labels.numpy(), labels])
        else:
            all_images, all_labels = images, labels

        per_class = REPLAY_BUFFER_SIZE // len(seen_classes)
        kept_images = []
        kept_labels = []

        for digit in sorted(seen_classes):
            indices = np.where(all_labels == digit)[0]
            take = min(per_class, len(indices))
            selected = self.rng.choice(indices, size=take, replace=False)

            kept_images.append(torch.tensor(all_images[selected], dtype=torch.float32))
            kept_labels.append(torch.tensor(all_labels[selected], dtype=torch.long))

        self.images = torch.cat(kept_images)
        self.labels = torch.cat(kept_labels)


def train_replay(task_arrays, loaders):
    set_seed()
    model = MNISTNetwork()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()
    buffer = ReplayBuffer()
    seen_classes = set()

    for task_id in range(1, 6):
        for _ in range(EPOCHS):
            model.train()
            for current_images, current_labels in loaders[task_id]:
                replay_images, replay_labels = buffer.sample()

                if replay_images is not None:
                    images = torch.cat([current_images, replay_images], dim=0)
                    labels = torch.cat([current_labels, replay_labels], dim=0)
                else:
                    images = current_images
                    labels = current_labels

                optimizer.zero_grad()
                loss = criterion(model(images), labels)
                loss.backward()
                optimizer.step()

        seen_classes.update(TASKS[task_id])
        buffer.update(task_arrays[task_id], seen_classes)

    return model


def prepare_models(repo_root: Path, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    task_arrays, loaders = load_training_data(repo_root)

    print("Training Naive Sequential...")
    naive = train_naive(loaders)
    torch.save(naive.state_dict(), output_dir / "naive_sequential_mnist.pt")
    print("Saved Naive Sequential model.")

    print("Training EWC...")
    ewc = train_ewc(loaders)
    torch.save(ewc.state_dict(), output_dir / "ewc_mnist.pt")
    print("Saved EWC model.")

    print("Training Experience Replay...")
    replay = train_replay(task_arrays, loaders)
    torch.save(replay.state_dict(), output_dir / "experience_replay_mnist.pt")
    print("Saved Experience Replay model.")

    print(f"All model checkpoints saved to {output_dir}.")
