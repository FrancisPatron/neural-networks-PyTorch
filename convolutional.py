import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from time import time
import matplotlib.pyplot as plt


class Network1(nn.Module):
    def __init__(self, name=""):
        super(Network1, self).__init__()
        self.name = name
        # simple convolutional block with ReLU activation
        self.conv_block = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=5, padding=2),
            nn.ReLU(),
        )
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(16 * 28 * 28, 10)

    def forward(self, x):
        x = self.conv_block(x)
        x = self.flatten(x)
        x = self.fc(x)
        return x


class Network2(nn.Module):
    def __init__(self, name=""):
        super(Network2, self).__init__()
        self.name = name
        # convolutional block with two convolutional layers and ReLU activations
        self.conv_block = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, padding=2),
            nn.ReLU(),
        )
        self.flatten = nn.Flatten()
        # fully connected block with a ReLU activation
        self.fc_block = nn.Sequential(
            nn.Linear(16 * 28 * 28, 84), nn.ReLU(), nn.Linear(84, 10)
        )

    def forward(self, x):
        x = self.conv_block(x)
        x = self.flatten(x)
        x = self.fc_block(x)
        return x


class Network3(nn.Module):
    def __init__(self, name=""):
        super(Network3, self).__init__()
        self.name = name
        # convolutional block with batch normalization and max pooling
        self.conv_block1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, padding=2),
            nn.BatchNorm2d(6),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )
        # another convolutional block with batch normalization and max pooling
        self.conv_block2 = nn.Sequential(
            nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )
        self.flatten = nn.Flatten()
        # connected block with multiple ReLU activations
        self.fc_block = nn.Sequential(
            nn.Linear(16 * 7 * 7, 120),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, 10),
        )

    def forward(self, x):
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.flatten(x)
        x = self.fc_block(x)
        return x


class Bono(nn.Module):
    def __init__(self, name=""):
        super(Bono, self).__init__()
        self.name = name
        # first convolutional block with batch normalization and ReLU activations
        self.conv_block1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )
        # second convolutional block with batch normalization and ReLU activations
        self.conv_block2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )

        # pooling and dropout layers
        self.maxpool = nn.MaxPool2d(2)
        self.dropout = nn.Dropout(p=0.5)
        self.flatten = nn.Flatten()
        # fully connected layers with ReLU activations
        self.fc1 = nn.Linear(64 * 14 * 14, 128)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.maxpool(x)
        x = self.dropout(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


def train(model, device, epochs=20):
    loss_fn = nn.CrossEntropyLoss()
    opt = optim.SGD(model.parameters(), lr=0.01)
    model.train()
    print(f"started training for {model.name}")
    losses = []
    stime = time()
    for epoch in range(epochs):
        running_loss = 0.0
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            opt.zero_grad()
            output = model(data)
            loss = loss_fn(output, target)
            loss.backward()
            opt.step()
            running_loss += loss.item()
        avg_loss = running_loss / (batch_idx + 1)
        losses.append(running_loss)
    etime = time()
    total_time = etime - stime
    # print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss.item()}")
    print(f"finished training for {model.name} in {total_time:.2f}s")
    return losses


def test(model, test_loader, device):
    correct = 0
    total = 0
    test_loss = 0
    model.eval()
    loss_fn = nn.CrossEntropyLoss()
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
            loss = loss_fn(output, target)
            test_loss += loss.item()
    test_loss /= len(test_loader)
    print(
        f"Test result for {model.name} CEL: {test_loss:.4f} Accuracy: {100 * correct / total}%"
    )


if __name__ == "__main__":
    # data
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    )

    train_dataset = datasets.MNIST(
        root="./data", train=True, download=True, transform=transform
    )
    test_dataset = datasets.MNIST(
        root="./data", train=False, download=True, transform=transform
    )

    train_loader = DataLoader(
        train_dataset, batch_size=128, shuffle=True, num_workers=16
    )
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using {device}")

    nn1 = Network1(name="Network 1").to(device)
    nn2 = Network2(name="Network 2").to(device)
    nn3 = Network3(name="Network 3").to(device)
    bono_nn = Bono(name="Bono NN").to(device)

    nn1_loss = train(nn1, device)
    nn2_loss = train(nn2, device)
    nn3_loss = train(nn3, device)
    bono_nn_loss = train(bono_nn, device)

    plt.plot(nn1_loss, "r", label=nn1.name)
    plt.plot(nn2_loss, "b", label=nn2.name)
    plt.plot(nn3_loss, "g", label=nn3.name)
    plt.plot(bono_nn_loss, "purple", label=bono_nn.name)
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
    test(bono_nn, test_loader, device)
