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

class RNNModelScratch:
    def __init__(self,vocab_size,num_hiddens,device):
        self.vocab_size = vocab_size
        self.num_hiddens = num_hiddens
        self.params = self.get_params(vocab_size,num_hiddens,device)

    def __call__(self, X, state):
        inputs = f.one_hot(X.T,self.vocab_size).float()
        return self.rnn(inputs,state,self.params)

    def begin_state(self, batch_size, device):
        return self.init_rnn_state(batch_size,self.num_hiddens,device)

    ## init parameters of rnn model
    def get_params(self,vocab_size,num_hiddens,device):
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

    ## init hidden state of rnn model
    def init_rnn_state(self,batch_size,num_hiddens,device):
        h=torch.zeros(
            (batch_size,num_hiddens),
            device=device
        )
        return (h,)

    ## front propagation
    def rnn(self,input,state,params):
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
        logits=torch.cat(
            outputs,
            dim=0
        )
        return logits,(h,)

net=RNNModelScratch(vocab_size,num_hiddens,device)

@torch.no_grad()
def predict(prefix,num_preds,net,vocab,device):
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

test=predict(
    prefix="time traveller ",
    num_preds=10,
    net=net,
    vocab=vocab,
    device=device
)
print(test)