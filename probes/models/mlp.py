"""The paper's MLP: one CPU training loop. Callers own the splits."""
from copy import deepcopy
from dataclasses import dataclass

import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
from torch import nn


@dataclass(frozen=True)
class MLPConfig:
    """One `mlp` block of the settings file. Every field is required."""
    standardize: bool
    hidden: tuple[int, ...]
    dropout: float
    batch_size: int
    epochs: int
    patience: int
    learning_rate: float
    weight_decay: float
    validation_fraction: float

    @classmethod
    def from_settings(cls, settings):
        return cls(**{**settings, "hidden": tuple(settings["hidden"])})


def fit_predict(X, y, train, validation, test, *, seed, config):
    """Fit one MLP, keep the checkpoint with the lowest validation loss, and return the
    held-out labels and probabilities for class 1."""
    torch.set_num_threads(1)
    torch.manual_seed(seed)
    layers = []
    width = X.shape[1]
    for hidden in config.hidden:
        layers.extend([nn.Linear(width, hidden), nn.ReLU(), nn.Dropout(config.dropout)])
        width = hidden
    model = nn.Sequential(*layers, nn.Linear(width, 2))
    X_train, X_val, X_test = X[train], X[validation], X[test]
    if config.standardize:
        scaler = StandardScaler().fit(X_train)
        X_train, X_val, X_test = (scaler.transform(a) for a in (X_train, X_val, X_test))
    x, v, t = (torch.as_tensor(a, dtype=torch.float32) for a in (X_train, X_val, X_test))
    y_train, y_val = (torch.as_tensor(y[i], dtype=torch.long) for i in (train, validation))
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    criterion = nn.CrossEntropyLoss()
    best = -np.inf
    checkpoint = None
    for epoch in range(config.epochs):
        generator = torch.Generator().manual_seed(seed * 100_000 + epoch)
        order = torch.randperm(len(x), generator=generator)
        model.train()
        for batch in order.split(config.batch_size):
            optimizer.zero_grad()
            criterion(model(x[batch]), y_train[batch]).backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            score = -criterion(model(v), y_val).item()
        if score > best:
            best = score
            checkpoint = deepcopy(model.state_dict())
            best_epoch = epoch
        if checkpoint is None:
            raise RuntimeError("Non-finite validation loss")
        if epoch - best_epoch >= config.patience:
            break
    model.load_state_dict(checkpoint)
    with torch.no_grad():
        logits = model(t)
        return logits.argmax(1).numpy(), logits.softmax(1)[:, 1].numpy()
