import torch
from torch import nn
from torch.utils.data import TensorDataset,DataLoader
import matplotlib
matplotlib.use('Agg')  # 使用非交互式绘图后端
import matplotlib.pyplot as plt

## make dataset
T=1000
time=torch.arange(1,T+1,dtype=torch.float32)
x=torch.sin(0.01*time)+torch.normal(0,0.02,(T,))

tau=4
features=torch.zeros(T-tau,tau)
for i in range(tau):
    features[:,i]=x[i:T-tau+i]
labels=x[tau:].reshape(-1,1)

## declare train and eval dataset
batch_size,n_train=16,600
train_dataset=TensorDataset(
    features[:n_train],
    labels[:n_train]
)
train_iter=DataLoader(
    dataset=train_dataset,
    batch_size=batch_size,
    shuffle=True
)

## make MLP
def init_weights(m):
    if type(m)==nn.Linear:
        nn.init.xavier_uniform_(m.weight)

def get_net():
    net=nn.Sequential(
        nn.Linear(4,10),
        nn.ReLU(),
        nn.Linear(10,1)
    )
    net.apply(init_weights)
    return net

loss=nn.MSELoss(reduction='sum')

def train_net(net,train_iter,loss,epochs,lr):
    trainer=torch.optim.Adam(net.parameters(),lr=lr)
    for epoch in range(epochs):
        l=0
        for X,y in train_iter:
            trainer.zero_grad()
            l=loss(net(X),y)
            l.backward()
            trainer.step()
        print(f"epoch:{epoch+1},loss:{l}")

net=get_net()
train_net(net,train_iter,loss,10,0.01)

## single-step predict
net.eval()
with torch.no_grad():
    onestep_preds = net(features)

plt.figure(figsize=(6, 3))

# 原始数据
plt.plot(
    time.numpy(),
    x.numpy(),
    label='data'
)
# 模型预测数据
plt.plot(
    time[tau:].numpy(),
    onestep_preds.squeeze(-1).numpy(),
    label='1-step preds'
)
plt.xlabel('time')
plt.ylabel('x')
plt.xlim(1, 1000)
plt.legend()

plt.tight_layout()

# 保存图片
plt.savefig('mlp_single_step_predictions.png', dpi=150)
plt.close()

## multi-step prediction
multistep_preds = torch.zeros(T)
multistep_preds[: n_train + tau] = x[: n_train + tau]
net.eval()
with torch.no_grad():
    for i in range(n_train + tau, T):
        multistep_preds[i] = net(
            multistep_preds[i - tau:i].reshape((1, -1))
        )

plt.figure(figsize=(6, 3))
# 原始数据
plt.plot(
    time.numpy(),
    x.numpy(),
    label='data'
)
# 模型预测数据
plt.plot(
    time[n_train + tau:].numpy(),
    multistep_preds[n_train + tau:].numpy(),
    label='multistep predictions'
)
plt.xlabel('time')
plt.ylabel('x')
plt.xlim(1, )
plt.legend()

plt.tight_layout()

# 保存图片
plt.savefig('mlp_multi_step_predictions.png', dpi=150)
plt.close()


## k-step prediction
steps = (1, 4, 16, 64)
max_steps = max(steps)
# 创建用于 k-step 预测的特征矩阵
num_samples = T - tau - max_steps + 1
k_features = torch.zeros(
    (num_samples, tau + max_steps)
)
# 1. 填充真实的历史数据
for i in range(tau):
    k_features[:, i] = x[i:i + num_samples]
# 2. 递归预测未来 64 个时间步
net.eval()
with torch.no_grad():
    for i in range(tau, tau + max_steps):
        k_features[:, i] = net(
            k_features[:, i - tau:i]
        ).reshape(-1)

plt.figure(figsize=(8, 4))
for k in steps:
    t = time[tau + k - 1:T - max_steps + k]
    preds = k_features[:, tau + k - 1]
    plt.plot(
        t.numpy(),
        preds.numpy(),
        label=f'{k}-step preds'
    )

plt.xlabel('time')
plt.ylabel('x')
plt.xlim(5, 1000)
plt.legend()
plt.tight_layout()

plt.savefig('mlp_k_step_predictions.png', dpi=150)
plt.close()