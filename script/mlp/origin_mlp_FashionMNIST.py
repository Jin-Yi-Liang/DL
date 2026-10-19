## mlp_fashion_mnist
import torch
import torch.nn as nn
import torchvision
from torch.utils.data import DataLoader
from torchvision import transforms
import time

start_tiime=time.perf_counter()

epoch_size=5
learning_rate=0.01
batch_size=256
input_size=28*28
out_size=10
hidden_size=256

device=torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)
print(f"device:{device}\n")

## get train and test data
transform=transforms.ToTensor()
train_dataset=torchvision.datasets.FashionMNIST(
    root="./data",
    train=True,
    transform=transform,
    download=True
)
train_loader=DataLoader(
    dataset=train_dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=0
)

test_dataset=torchvision.datasets.FashionMNIST(
    root="./data",
    train=False,
    transform=transform,
    download=True
)
test_loader=DataLoader(
    dataset=test_dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=0
)

## initialize model
net=nn.Sequential(
    nn.Flatten(),
    nn.Linear(input_size,hidden_size),
    nn.ReLU(),
    nn.Linear(hidden_size,out_size)
)

nn.init.normal_(
    net[1].weight,
    mean=0,
    std=0.01
)
nn.init.zeros_(
    net[1].bias
)

nn.init.normal_(
    net[3].weight,
    mean=0,
    std=0.01
)
nn.init.zeros_(
    net[3].bias
)

net=net.to(device)
print(f"model shape:{net}")

## prepare loss function
loss_func=nn.CrossEntropyLoss()

## prepare optimizer
optimizer=torch.optim.SGD(
    params=net.parameters(),
    lr=learning_rate)

## run train
def run_one_epoch():
    total_loss=0
    total_correct=0
    total_sample=0

    for X,y in train_loader:
        X=X.to(device)
        y=y.to(device)

        y_hat=net(X)
        loss=loss_func(y_hat,y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_size=y.size(0)
        total_loss+=batch_size*loss.item()
        predictions=y_hat.argmax(dim=1)
        correct=(predictions==y).sum().item()
        total_correct+=correct
        total_sample+=batch_size

    epoch_loss=total_loss/total_sample
    epoch_correct=total_correct/total_sample
    return epoch_loss,epoch_correct

## run eval
def run_eval():
    total_loss=0
    total_correct=0
    total_sample=0

    with torch.no_grad():
        for X,y in test_loader:
            X=X.to(device)
            y=y.to(device)

            y_hat=net(X)
            loss=loss_func(y_hat,y)

            batch_size=y.size(0)
            total_loss+=loss.item()*batch_size
            prediction=y_hat.argmax(dim=1)
            correct=(prediction==y).sum().item()
            total_correct+=correct
            total_sample+=batch_size

        test_loss=total_loss/total_sample
        test_correct=total_correct/total_sample
        return test_loss,test_correct


## main 
for epoch in range(epoch_size):
    train_loss_asset=[]
    train_correct_asset=[]
    test_loss_asset=[]
    test_correct_asset=[]

    train_loss,train_correct=run_one_epoch()
    test_loss,test_correct=run_eval()
    
    train_loss_asset.append(train_loss)
    train_correct_asset.append(train_correct)
    test_loss_asset.append(test_loss)
    test_correct_asset.append(test_correct)

    print(
        f"Epoch:{epoch+1}\n"
        f"train loss:{train_loss:.4f}\n"
        f"train correct:{train_correct:.4f}\n"
        f"test loss:{test_loss:.4f}\n"
        f"test correct:{test_correct:.4f}\n"
    )

end_time=time.perf_counter()
print(f"Time cost:{end_time-start_tiime}")