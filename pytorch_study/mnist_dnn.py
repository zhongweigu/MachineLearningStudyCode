import torch
from torchvision import datasets, transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


# print(torch.__version__)

use_cuda = torch.cuda.is_available()
# print(use_cuda)

if use_cuda:
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

ds1 = datasets.MNIST('./data', train=True, download=True, transform=transform)
ds2 = datasets.MNIST('./data', train=False, download=True, transform=transform)

train_loader = torch.utils.data.DataLoader(ds1, batch_size=128, shuffle=True)
test_loader = torch.utils.data.DataLoader(ds2, batch_size=1000)

'''
for batch_idx, data in enumerate(train_loader,0):
    inputs, targets = data
    x = inputs.view(-1,28*28)
    x_std = x.std().item()
    x_mean = x.mean().item()

print('均值mean是:'+str(x_mean))
print('标准差std是:'+str(x_std))
'''

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(784, 128)
        self.dropout = nn.Dropout(0.2)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output

model = Net().to(device)


def train_step(data, target, model, optimizer):
    optimizer.zero_grad()
    output = model(data)
    loss = F.nll_loss(output, target)
    loss.backward()
    optimizer.step()
    return loss


def test_step(data, target, model, test_loss, correct):
    output = model(data)
    test_loss += F.nll_loss(output, target, reduction='sum').item()
    pred = output.argmax(1, keepdim=True)
    correct += pred.eq(target.view_as(pred)).sum().item()
    return test_loss, correct

EPOCHS = 5
optimizer = optim.Adam(model.parameters(), lr=0.001)

for epoch in range(EPOCHS):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        loss = train_step(data, target, model, optimizer)
        if batch_idx % 10 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss:{:.6f}'.format(epoch, batch_idx * len(data),
                  len(train_loader.dataset) ,
                  100 * batch_idx / len(train_loader), loss.item()))

    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            test_loss, correct = test_step(data, target, model, test_loss, correct)
    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(test_loss, correct, len(test_loader.dataset),
                                                                                 100 * correct / len(test_loader.dataset)))