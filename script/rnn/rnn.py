import torch
import torch.nn as nn
import torch.nn.functional as f

import math
import time
import random

import matplotlib.pyplot as plt

from time_machine_dataloader import load_time_machine_dataloader

## super parameters
batch_size=32   
num_step=35
num_hiddens=512

learning_rate=1
num_epoch=1000

random.seed(42)
torch.manual_seed(42)

print("batch size:",batch_size)
print("num step:",num_step)
print("num hiddens:",num_hiddens)
print("learning rate:",learning_rate)
print("num epoch:",num_epoch)

## move to proper device
device=torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)
print("current device:",device)

## get dataloader and vocab
train_loader,vocab=load_time_machine_dataloader(batch_size,num_step)
vocab_size=len(vocab)
print("vocab size:",vocab_size)

x,y=next(iter(train_loader))
x=x.to(device)
y=y.to(device)

class RNNModelScratch(nn.Module):
    def __init__(self,vocab_size,num_hiddens,device):
        super(RNNModelScratch, self).__init__()
        self.vocab_size = vocab_size
        self.num_hiddens = num_hiddens
        self.w_xh = nn.Parameter(torch.randn(vocab_size,num_hiddens,device=device)*0.01)
        self.w_hh = nn.Parameter(torch.randn(num_hiddens,num_hiddens,device=device)*0.01)
        self.b_h = nn.Parameter(torch.zeros(num_hiddens,device=device))
        self.w_hq = nn.Parameter(torch.randn(num_hiddens,vocab_size,device=device)*0.01)
        self.b_q = nn.Parameter(torch.zeros(vocab_size,device=device))

    def forward(self, X, state):
        inputs = f.one_hot(X.T,self.vocab_size).float()
        return self.rnn(inputs,state)

    def begin_state(self, batch_size, device):
        return self.init_rnn_state(batch_size,self.num_hiddens,device)

    ## init hidden state of rnn model
    def init_rnn_state(self,batch_size,num_hiddens,device):
        h=torch.zeros(
            (batch_size,num_hiddens),
            device=device
        )
        return (h,)

    ## front propagation
    def rnn(self,input,state):
        h=state[0]
        outputs=[]
        for x_t in input:
            h=torch.tanh(
                x_t@self.w_xh
                +h@self.w_hh
                +self.b_h
            )
            logits_t=h@self.w_hq+self.b_q
            outputs.append(logits_t)
        logits=torch.cat(
            outputs,
            dim=0
        )
        return logits,(h,)

net=RNNModelScratch(vocab_size,num_hiddens,device)

@torch.no_grad()
def predict(prefix,num_preds,net,vocab,device=device):
    net.eval()
    state=net.begin_state(batch_size=1,device=device)
    outputs=[vocab[prefix[0]]]

    def get_input():
        return torch.tensor(
            [[outputs[-1]]],
            dtype=torch.long,
            device=device
        )

    ## use the prefix to warm up the model
    for char in prefix[1:]:
        _,state=net(get_input(),state)
        outputs.append(vocab[char])

    ## predict the next tokens
    for _ in range(num_preds):
        logits,state=net(get_input(),state)
        next_token=logits.argmax(dim=1).item()
        outputs.append(next_token)
    
    return "".join(
        vocab.idx_to_token[i] for i in outputs
    )

def train_one_epoch(train_loader,net,optimizer,use_random_iter=False,device=device):
    net.train()
    total_loss=0
    total_tokens=0
    state=None
    start_time=time.perf_counter()
    for X,Y in train_loader:
        X=X.to(device)
        Y=Y.to(device)
        if state is None or use_random_iter:
            state=net.begin_state(batch_size=X.shape[0],device=device)
        else:
            state=tuple(s.detach() for s in state)
        optimizer.zero_grad()
        logits,state=net(X,state)
        y=Y.T.reshape(-1)
        loss=f.cross_entropy(logits,y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(),max_norm=1.0)
        optimizer.step()
        num_token=y.numel()
        total_loss+=loss.item()*num_token
        total_tokens+=num_token
    avg_loss=total_loss/total_tokens
    ppl=math.exp(avg_loss)
    if device.type=="cuda":
        torch.cuda.synchronize()
    elapsed=time.perf_counter()-start_time
    speed=total_tokens/elapsed
    return avg_loss,ppl,speed

optimizer=torch.optim.SGD(
    net.parameters(),
    lr=learning_rate
)

def train(train_loader,net,vocab,optimizer,device,num_epoches=500):
    history=[]
    for epoch in range(num_epoches):
        avg_loss,ppl,speed=train_one_epoch(train_loader,net,optimizer)
        history.append(ppl)
        if (epoch+1)%50 == 0:
            print(f"===== epoch: {epoch+1} =====")
            print(f"loss: {avg_loss}")
            print(f"ppl: {ppl}")
            print(f"speed: {speed} token/s")
            print(predict("time traveller ",200,net,vocab,device))
    return history

history=train(train_loader,net,vocab,optimizer,device,num_epoches=num_epoch)

def plot_plexity(history):
    epoches=range(1,len(history)+1)
    plt.figure(figsize=(8,5))
    plt.plot(epoches,history,label="Training ppl")
    plt.xlabel("Epoch")
    plt.ylabel("Perplexity")
    plt.title("RNN training")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("rnn_train_ppl.png",dpi=150)
plot_plexity(history)