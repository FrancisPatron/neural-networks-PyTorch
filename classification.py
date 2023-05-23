import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from time import time
import matplotlib.pyplot as plt


class Network1(nn.Module):  # 4 - layer network
    def __init__(self, num_classes, name=""):
        super(Network1, self).__init__()
        self.name = name
        self.fc1 = nn.Linear(28 * 28, 20)
        self.fc2 = nn.Linear(20, 50)
        self.fc3 = nn.Linear(50, 20)
        self.fc4 = nn.Linear(20, num_classes)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = torch.flatten(x, 1)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.fc4(x)
        return nn.functional.log_softmax(x, dim=1)


class Network2(nn.Module):  # 6 - layer network
    def __init__(self, num_classes, name=""):
        super(Network2, self).__init__()
        self.name = name
        self.fc1 = nn.Linear(28 * 28, 10)
        self.fc2 = nn.Linear(10, 20)
        self.fc3 = nn.Linear(20, 30)
        self.fc4 = nn.Linear(30, 20)
        self.fc5 = nn.Linear(20, 10)
        self.fc6 = nn.Linear(10, num_classes)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = torch.flatten(x, 1)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.relu(self.fc4(x))
        x = self.relu(self.fc5(x))
        x = self.fc6(x)
        return nn.functional.log_softmax(x, dim=1)


class Network3(nn.Module):  # 6 - layer network
    def __init__(self, num_classes, name=""):
        super(Network3, self).__init__()
        self.name = name
        self.fc1 = nn.Linear(28 * 28, 10)
        self.fc2 = nn.Linear(10, 40)
        self.fc3 = nn.Linear(40, 70)
        self.fc4 = nn.Linear(70, 40)
        self.fc5 = nn.Linear(40, 10)
        self.fc6 = nn.Linear(10, num_classes)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = torch.flatten(x, 1)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.relu(self.fc4(x))
        x = self.relu(self.fc5(x))
        x = self.fc6(x)
        return nn.functional.log_softmax(x, dim=1)


def train(model, train_loader, device, epochs=20):
    losses = []
    model.train()
    loss_fn = nn.CrossEntropyLoss()
    opt = optim.Adam(model.parameters(), lr=0.01)
    print(f"starting training for {model.name} with {device}")
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
    print(f"done training {model.name}  in {etime - stime:.2f}s")
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
    # Define Hyper-parameters
    num_classes = 10
    batch_size = 256
    learning_rate = 0.01

    # MNIST dataset
    train_dataset = torchvision.datasets.MNIST(
        root="./data", train=True, transform=transforms.ToTensor(), download=True
    )
    test_dataset = torchvision.datasets.MNIST(
        root="./data", train=False, transform=transforms.ToTensor()
    )

    # Data loader
    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset, batch_size=batch_size, shuffle=True
    )
    test_loader = torch.utils.data.DataLoader(
        dataset=test_dataset, batch_size=batch_size, shuffle=True
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using {device}")

    nn1 = Network1(num_classes, name="nn1").to(device)
    nn2 = Network2(num_classes, name="nn2").to(device)
    nn3 = Network3(num_classes, name="nn3").to(device)

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
