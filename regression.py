import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from time import time
import matplotlib.pyplot as plt


class Network1(nn.Module):
    def __init__(self, input_size, name=""):
        self.name = name
        super(Network1, self).__init__()
        self.layer = nn.Linear(input_size, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.layer(x)
        x = self.relu(x)
        return x


class Network2(nn.Module):
    def __init__(self, input_size, name=""):
        super(Network2, self).__init__()
        self.name = name
        self.model = nn.Sequential(
            nn.Linear(input_size, 10),
            nn.ReLU(),
            nn.Linear(10, 20),
            nn.ReLU(),
            nn.Linear(20, 10),
            nn.ReLU(),
            nn.Linear(10, 1),
        )

    def forward(self, x):
        return self.model(x)


class Network3(nn.Module):
    def __init__(self, input_size, name=""):
        super(Network3, self).__init__()
        self.name = name
        self.model = nn.Sequential(
            nn.Linear(input_size, 10),
            nn.ReLU(),
            nn.Linear(10, 20),
            nn.ReLU(),
            nn.Linear(20, 30),
            nn.ReLU(),
            nn.Linear(30, 20),
            nn.ReLU(),
            nn.Linear(20, 1),
        )

    def forward(self, x):
        return self.model(x)


def train(model, train_loader, device, epochs=20):
    losses = []
    model.train()
    loss_fn = nn.MSELoss()
    opt = optim.Adam(model.parameters(), lr=0.01)
    print(f"starting training for {model.name}")
    stime = time()
    for epoch in range(epochs):
        running_loss = 0.0
        for i, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = loss_fn(outputs, targets)
            opt.zero_grad()
            loss.backward()
            opt.step()
            running_loss += loss.item()
        avg_loss = running_loss / (i + 1)
        losses.append(avg_loss)
        # print(f"    {model.name} [{epoch + 1}/{epochs}], Loss: {avg_loss:.4f}")
    etime = time()
    print(f"done training {model.name} in {etime - stime:.2f}s")
    return losses


def test(model, test_loader, device, threshold=1.0):
    model.eval()
    loss_fn = nn.MSELoss()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = loss_fn(outputs, labels)
            test_loss += loss.item()
            total += labels.size(0)
            correct += torch.sum(torch.abs(outputs - labels) <= threshold).item()
    test_loss /= len(test_loader)
    accuracy = 100 * correct / total
    print(
        f"Test results for {model.name} MSE: {test_loss:.4f}, Accuracy: {accuracy:.2f}% (within ±{threshold})\n"
    )


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using {device}")

    # Data loading and preprocessing
    file_path = "./data/auto-mpg/auto-mpg.data"
    url = "http://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data"
    column_names = [
        "MPG",
        "Cylinders",
        "Displacement",
        "Horsepower",
        "Weight",
        "Acceleration",
        "Model Year",
        "Origin",
    ]

    try:
        dataset = pd.read_csv(
            file_path,
            names=column_names,
            na_values="?",
            comment="\t",
            sep=" ",
            skipinitialspace=True,
        )
    except:
        print(f"error while reading {file_path}, using {url} instead")
        dataset = pd.read_csv(
            url,
            names=column_names,
            na_values="?",
            comment="\t",
            sep=" ",
            skipinitialspace=True,
        )

    # Data cleaning
    dataset = dataset.dropna()
    dataset["Origin"] = dataset["Origin"].map({1: "USA", 2: "Europe", 3: "Japan"})
    dataset = pd.get_dummies(dataset, columns=["Origin"], prefix="", prefix_sep="")

    # Convert the categorical columns to integer type
    dataset["USA"] = dataset["USA"].astype(int)
    dataset["Europe"] = dataset["Europe"].astype(int)
    dataset["Japan"] = dataset["Japan"].astype(int)

    # Splitting the data into train/test sets
    train_dataset = dataset.sample(frac=0.8, random_state=0)
    test_dataset = dataset.drop(train_dataset.index)

    # Separate features & labels (MPG)
    train_features = train_dataset.copy()
    test_features = test_dataset.copy()
    train_labels = train_features.pop("MPG")
    test_labels = test_features.pop("MPG")

    # Normalize data using PyTorch
    train_features_values = torch.Tensor(train_features.values.astype(float))
    test_features_values = torch.Tensor(test_features.values.astype(float))

    mean = train_features_values.mean(dim=0)
    std = train_features_values.std(dim=0)

    train_features = (train_features_values - mean) / std
    test_features = (test_features_values - mean) / std

    # Convert to PyTorch tensors
    X_train_tensor = torch.Tensor(train_features)
    y_train_tensor = torch.Tensor(train_labels.values).view(-1, 1)
    X_test_tensor = torch.Tensor(test_features)
    y_test_tensor = torch.Tensor(test_labels.values).view(-1, 1)

    # Create data loaders
    train_dataset = torch.utils.data.TensorDataset(X_train_tensor, y_train_tensor)
    test_dataset = torch.utils.data.TensorDataset(X_test_tensor, y_test_tensor)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=16, shuffle=True
    )
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=16, shuffle=True)

    input_size = X_train_tensor.shape[1]
    nn1 = Network1(input_size, name="nn1").to(device)
    nn2 = Network2(input_size, name="nn2").to(device)
    nn3 = Network3(input_size, name="nn3").to(device)

    nn1_loss = train(nn1, train_loader, device)
    nn2_loss = train(nn2, train_loader, device)
    nn3_loss = train(nn3, train_loader, device)

    plt.plot(nn1_loss, "r", label="Network 1")
    plt.plot(nn2_loss, "b", label="Network 2")
    plt.plot(nn3_loss, "g", label="Network 3")
    plt.tight_layout()
    plt.grid(True, color="y")
    plt.xlabel("Epochs/Iterations")
    plt.ylabel("Loss")
    plt.title("Loss Comparison of Networks")
    plt.legend()
    plt.show()

    test(nn1, test_loader, device)
    test(nn2, test_loader, device)
    test(nn3, test_loader, device)
