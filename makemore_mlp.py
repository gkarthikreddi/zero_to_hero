import torch
import torch.nn.functional as F
import random


# read in all the words
words = open("names.txt", 'r').read().splitlines()

# mapping characters to intergers and vise versa
chars = sorted(list(set(''.join(words))))
stoi = {s: i+1 for i, s in enumerate(chars)}
stoi['.'] = 0
itos = {i: s for s, i in stoi.items()}


# build the dataset
def build_dataset(words):
    block_size = 3  # how many characters we take to predict the next word
    X, Y = [], []

    for w in words:
        context = [0] * block_size
        for ch in w + '.':
            ix = stoi[ch]
            X.append(context)
            Y.append(ix)
            context = context[1:] + [ix]

    X = torch.tensor(X)
    Y = torch.tensor(Y)
    return X, Y


random.shuffle(words)
n1 = int(0.8*len(words))
n2 = int(0.9*len(words))

Xtr, Ytr = build_dataset(words[:n1])
Xdev, Ydev = build_dataset(words[n1:n2])
Xte, Yte = build_dataset(words[n2:])

C = torch.randn((27, 10))

"""
F.one_hot(torch.tensor(5), num_classes=27).float() @ C == C[5]
so we use C[5] instead of one_hot method, cause it's simple
"""

# emb = C[X]

W1 = torch.randn((30, 200))
b1 = torch.randn(200)

"""
We can't multiply emb @ W1 + b1 as the dimensions of emb is (32, 3, 2)
First we need to concatenate emb elements to dimensions (32, 6)
i.e torch.cat([emb[:, 0, :], emb[:, 1, :], emb[:, 2, :])
The above aproach is unscalable so we use "torch.unbind"
torch.cat(torch.unbind(emb, 1), 1)

But again this in every infecient so we use "torch.view"
"""
# h = torch.tanh(emb.view(-1, 6) @ W1 + b1)

W2 = torch.randn((200, 27))
b2 = torch.randn(27)
# logits = h @ W2 + b2

"""
count = logits.exp()
prob = count / count.sum(1, keepdims=True)
loss = -prob[torch.arrange(len(Y)), Y].log().mean()

The above can written be in an efficient way using "cross_entropy" function
"""
# loss = F.corss_entropy(logits, Y)
parameters = [C, W1, b1, W2, b2]

for p in parameters:
    p.requires_grad = True

# learning rate
lre = torch.linspace(-3, 0, 1000)
lrs = 100*lre

# Training
for i in range(200000):
    # minibatch construct
    ix = torch.randint(0, Xtr.shape[0], (32,))

    # forwardpass
    emb = C[Xtr[ix]]
    h = torch.tanh(emb.view(-1, 30) @ W1 + b1)
    logits = h @ W2 + b2
    loss = F.cross_entropy(logits, Ytr[ix])

    # backwardpass
    for p in parameters:
        p.grad = None
    loss.backward()

    # choosing the correct learning rate
    # lr = lrs[i]
    lr = 0.1 if i < 100000 else 0.01

    # update
    for p in parameters:
        p.data += -lr * p.grad
print(f"Training loss:{loss}")

# Calculating dev loss
emb = C[Xdev]
h = torch.tanh(emb.view(-1, 30) @ W1 + b1)
logits = h @ W2 + b2
loss = F.cross_entropy(logits, Ydev)
print(f"DEV loss:{loss}")

# sample from the model
for _ in range(10):
    out = []
    context = [0] * 3  # block_size  # initialize with all ...
    while True:
        emb = C[torch.tensor([context])]  # (1,block_size,d)
        h = torch.tanh(emb.view(1, -1) @ W1 + b1)
        logits = h @ W2 + b2
        probs = F.softmax(logits, dim=1)
        ix = torch.multinomial(probs, num_samples=1).item()
        context = context[1:] + [ix]
        out.append(ix)
        if ix == 0:
            break

    print(''.join(itos[i] for i in out))
