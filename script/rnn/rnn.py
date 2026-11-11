import torch
import torch.nn.functional as f

import math
import time

import matplotlib.pyplot as plt

from time_machine_dataloader import load_time_machine_dataloader

## super parameters
batch_size=32
num_step=35
num_hiddens=512

learning_rate=1.0
num_epoch=500
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

## one-hot encoding
x_onehot=f.one_hot(
    x.T,
    num_classes=vocab_size
).type(torch.float32)

## init parameters of rnn model
def get_params(vocab_size,num_hiddens,device):
    num_inputs=vocab_size
    num_outputs=vocab_size

    def normal(shape):
        return torch.randn(
            shape,
            device=device
        )*0.01
    
    w_xh=normal((num_inputs,num_hiddens))
    w_hh=normal((num_hiddens,num_hiddens))
    b_h=torch.zeros(num_hiddens,device=device)
    w_hq=normal((num_hiddens,num_outputs))
    b_q=torch.zeros(num_outputs,device=device)

    params=[w_xh,w_hh,b_h,w_hq,b_q]

    for param in params:
        param.requires_grad_(True)

    return params

params=get_params(vocab_size,num_hiddens,device)
for param in params:
    print(param.shape)
total_params=sum(
    p.numel() for p in params
)
print("total params:",total_params)

## init hidden state of rnn model
def init_rnn_state(batch_size,num_hiddens,device):
    h=torch.zeros(
        (batch_size,num_hiddens),
        device=device
    )
    return (h,)
state=init_rnn_state(batch_size,num_hiddens,device)
print("state shape:",state[0].shape)
print("number of state elements:",len(state))

## front propagation
def rnn(input,state,params):
    print("===== front propagation =====")
    w_xh,w_hh,b_h,w_hq,b_q=params
    h=state[0]
    outputs=[]
    for x_t in input:
        h=torch.tanh(
            x_t@w_xh
            +h@w_hh
            +b_h
        )
        logits_t=h@w_hq+b_q
        outputs.append(logits_t)
    print("outputs length:",len(outputs))
    print("outputs[0] shape:",outputs[0].shape)

    logits=torch.cat(
        outputs,
        dim=0
    )
    return logits,(h,)

logits,state=rnn(x_onehot,state,params)
print("logits shape:",logits.shape)
print("hidden state shape:",state[0].shape)