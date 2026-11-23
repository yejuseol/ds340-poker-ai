
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from models.imitation_net import ImitationNet

DATA_DIR = Path("data/processed")
MODEL_PATH = Path("models/imitation.pt")

EPOCHS = 20
BATCH_SIZE = 64
LEARNING_RATE = 0.001

# Load prepared datasets
def load_data(split):
    X = np.load(DATA_DIR / f"X_{split}.npy")
    y = np.load(DATA_DIR / f"y_{split}.npy")

    return TensorDataset(
        torch.tensor(X, dtype=torch.float32),
        torch.tensor(y, dtype=torch.long)
    )

train_data = load_data("train")
val_data = load_data("val")
test_data = load_data("test")

train_loader = DataLoader(
    train_data, batch_size=BATCH_SIZE, shuffle=True
)
val_loader = DataLoader(
    val_data, batch_size=BATCH_SIZE
)
test_loader = DataLoader(
    test_data, batch_size=BATCH_SIZE
)

# Initialize neural network
torch.manual_seed(42)
model = ImitationNet(input_size=108, num_actions=4)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(
    model.parameters(), lr=LEARNING_RATE
)


def evaluate(loader):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for X, y in loader:
            outputs = model(X)
            predictions = outputs.argmax(dim=1)

            correct += (predictions == y).sum().item()
            total += len(y)

    return correct / total


# Compare against always predicting the most common action
train_labels = train_data.tensors[1].numpy()
test_labels = test_data.tensors[1].numpy()

majority_action = np.bincount(train_labels).argmax()
baseline_accuracy = np.mean(
    test_labels == majority_action
)

print("Majority-class test accuracy:",
      f"{baseline_accuracy:.3f}")

# Train network
best_val_accuracy = 0.0

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0.0

    for X, y in train_loader:
        optimizer.zero_grad()

        outputs = model(X)
        loss = criterion(outputs, y)

        loss.backward()
        optimizer.step()

        total_loss += loss.item() * len(y)

    train_loss = total_loss / len(train_data)
    val_accuracy = evaluate(val_loader)

    print(
        f"Epoch {epoch + 1}/{EPOCHS} | "
        f"Loss: {train_loss:.4f} | "
        f"Val Accuracy: {val_accuracy:.3f}"
    )

    if val_accuracy > best_val_accuracy:
        best_val_accuracy = val_accuracy
        MODEL_PATH.parent.mkdir(
            parents=True, exist_ok=True
        )
        torch.save(model.state_dict(), MODEL_PATH)

# Evaluate best saved model on unseen test data
model.load_state_dict(
    torch.load(MODEL_PATH, map_location="cpu",
               weights_only=True)
)

test_accuracy = evaluate(test_loader)

print("\n=== FINAL RESULTS ===")
print("Best validation accuracy:",
      f"{best_val_accuracy:.3f}")
print("Test accuracy:", f"{test_accuracy:.3f}")
print("Majority baseline:", f"{baseline_accuracy:.3f}")
print("Model saved to:", MODEL_PATH)
