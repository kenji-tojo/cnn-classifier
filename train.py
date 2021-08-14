import torch
import torch.optim
import torch.utils.data
from torchvision import datasets, transforms
import os

from model import Classifier

def _train_epoch(
    net,
    epoch,
    train_loader,
    optimizer,
    criterion,
    device,
    log_interval=100
    ):
    net.train()
    loss = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data = data.to(device)
        target = target.to(device)
        optimizer.zero_grad()
        output = net(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % log_interval == 0:
            print('Train epoch: {} [{}/{} ({:0f}%)]\tLoss: {:.6f}'.format(
                epoch,
                batch_idx * len(data),
                len(train_loader.dataset),
                100. * batch_idx / len(train_loader),
                loss.data.item()))

def _validate(
    net,
    loader,
    criterion,
    device,
    loss_vector,
    accuracy_vector
    ):
    net.eval()
    loss, correct = 0, 0
    for data, target in loader:
        data = data.to(device)
        target = target.to(device)
        output = net(data)
        loss += criterion(output, target).data.item()
        pred = output.data.max(1)[1]
        correct += pred.eq(target.data).cpu().sum()
    loss /= len(loader)
    accuracy = 100. * correct.to(torch.float32) / len(loader.dataset)
    loss_vector.append(loss)
    accuracy_vector.append(accuracy.item())
    print('Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        loss,
        correct,
        len(loader.dataset),
        accuracy))

def train(
    net,
    train_loader,
    validation_loader,
    optimizer,
    criterion,
    device,
    n_epochs,
    save_interval=10
    ):
    losst, acct = list(), list()
    lossv, accv = list(), list()
    for epoch in range(1, n_epochs+1):
        _train_epoch(
            net, epoch, train_loader, optimizer, criterion, device)
        print('\n***Checking loss/acc at the end of the epoch***\nTrain: ')
        _validate(
            net, train_loader, criterion, device, losst, acct)
        print('***Checking loss/acc at the end of the epoch***\nValidation: ')
        _validate(
            net, validation_loader, criterion, device, lossv, accv)
        if epoch % save_interval == 0:
            torch.save(net.state_dict(), 'results/weights.pt')
    return losst, acct, lossv, accv

def main(
    batch_size=100,
    lr=5e-3,
    weight_decay=8e-6,
    n_epochs=30
    ):
    os.makedirs('results', exist_ok=True)

    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    print('Using {}'.format(device))

    train_dataset = datasets.KMNIST('data', train=True, download=True, transform=transforms.ToTensor())
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    validation_dataset = datasets.KMNIST('data', train=False, transform=transforms.ToTensor())
    validation_loader = torch.utils.data.DataLoader(dataset=validation_dataset, batch_size=batch_size, shuffle=False)

    net = Classifier().to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = torch.nn.CrossEntropyLoss()

    losst, acct, lossv, accv = train(
        net, train_loader, validation_loader,
        optimizer, criterion, device, n_epochs)

if __name__ == '__main__':
    main()