import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import os

# hyper-params
batch_size = 100
lr = 0.001
epoch = 10
seed = 12345

# load data
train_data = datasets.FashionMNIST(root='./data', train=True, download=True, transform=transforms.ToTensor())
test_data = datasets.FashionMNIST(root='./data', train=False, download=True, transform=transforms.ToTensor())

train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

"""與 Tensorflow 不同，讀出圖片ToTensor後已經是 0~1 的數值，不需再除以 255"""

# Show the image
# plt.figure()
# plt.imshow(train_data[0][0].squeeze())
# plt.colorbar()
# plt.grid(False)
# plt.show()

def same_seeds(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


same_seeds(seed)


# define the network
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.BatchNorm1d(784),
            nn.Linear(784, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Linear(32, 10)
        )

    def forward(self, x):
        x = self.net(x)
        return x


# checking device
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device}")

model = Net().to(device)

# loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

best_acc = 0
for e in range(epoch):
    print(f"\nepoch: {e + 1}")
    # training
    model.train()
    for batch, (X, y) in enumerate(train_loader):
        pred = model(X)
        batch_loss = criterion(pred, y)

        optimizer.zero_grad()
        batch_loss.backward()
        optimizer.step()

        if (batch+1) % 10 == 0:
            print(f"\rloss: {batch_loss.item()} [{batch_size * (batch+1)}/{len(train_loader.dataset)}]", end='')

    # evaluating
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in test_loader:
            pred = model(X)
            pred_label = pred.argmax(dim=1)  # _, pred_label = torch.max(pred, dim=1)
            test_loss += criterion(pred, y).item()
            correct += (pred_label == y).sum().item()

    test_loss /= (len(test_loader.dataset) / batch_size)
    correct /= len(test_loader.dataset)
    print(f"\nTest Error:  Accuracy: {(100 * correct):>0.1f}%, Avg loss: {test_loss:>8f}")

    if correct > best_acc:
        best_acc = correct
        torch.save(model.state_dict(), "model.pth")
        print("Best Model Saved")


# use the best model to predict
saved_model = Net()
saved_model.load_state_dict(torch.load("model.pth"))

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

saved_model.eval()
with torch.no_grad():
    pred = saved_model(test_loader.dataset[12])
# test_loss, correct = 0, 0
# with torch.no_grad():
#     for X, y in test_loader:
#         pred = saved_model(X)
#         pred_label = pred.argmax(dim=1)  # _, pred_label = torch.max(pred, dim=1)
#         test_loss += criterion(pred, y).item()
#         correct += (pred_label == y).sum().item()
#
# test_loss /= (len(test_loader.dataset) / batch_size)
# correct /= len(test_loader.dataset)
print(f"\nBest Model:  Accuracy: {(100 * best_acc):>0.1f}% \n")
