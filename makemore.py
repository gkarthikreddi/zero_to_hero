import torch
import torch.nn.functional as F


words = open('names.txt', 'r').read().splitlines()

# Getting all the characters in the words and sorting them
chars = sorted(list(set(''.join(words))))

# Mapping characters to number to insert into array
stoi = {s:i+1 for i, s in enumerate(chars)}
stoi['.'] = 0

# Mapping integers to characters
itos = {i:s for s, i in stoi.items()}

""" Predicting without Neural Nets

N = torch.zeros((27, 27), dtype = torch.int32)

for w in words:
    chs = '.' + w + '.'
    for ch1, ch2 in zip(chs, chs[1:]):
        ix1 = stoi[ch1]
        ix2 = stoi[ch2]
        N[ix1, ix2] += 1

# Building a Probability array for the characters
P = (N+1).float()
P /= P.sum(1, keepdims=True) # Creates a [27, 1] tensor which contains the sums of each row and normalizes the row

for i in range(10): out = []
    ix = 0
    while True:
        #p = N[ix].float()
        #p /= p.sum()

        p = P[ix]
        ix = torch.multinomial(p, num_samples=1, replacement=True).item()
        out.append(itos[ix])
        if ix == 0:
            break
    print(''.join(out))
"""

# Create the training set of biograms(x,y)
xs, ys = [], []

for w in words:
    chs = '.' + w + '.'
    for ch1, ch2 in zip(chs, chs[1:]):
        ix1 = stoi[ch1]
        ix2 = stoi[ch2]
        xs.append(ix1)
        ys.append(ix2)

xs = torch.tensor(xs)
ys = torch.tensor(ys)
num = xs.nelement()

# Same thing Using Neural Networks

# Generating 27 random number as lists with normal distribution for Weigts
W = torch.randn(27, 27, requires_grad=True)

# Gradient Decend
for k in range(40):
# Farwordpass
    xenc = F.one_hot(xs, num_classes=27).float()
    logits = xenc @ W # here "@" does matrix multiplication and this is called log counts
    counts = logits.exp() # counts, equivalent to N (above)
    probs = counts / counts.sum(1, keepdims=True) # probabilties for nex characters
# The above two steps are called soft max

# Magic
    loss = -probs[torch.arange(num), ys].log().mean()
    print(f"{loss=}")

# Backwardpass
    W.grad = None
    loss.backward()

# Update
    W.data += -50 * W.grad

# Sample output from neural Nets
for i in range(10):
    out = []
    ix = 0
    while True:
        xenc = F.one_hot(torch.tensor([ix]), num_classes=27).float()
        logits = xenc @ W
        counts = logits.exp()
        p = counts / counts.sum(1, keepdims=True)
        ix = torch.multinomial(p, num_samples=1, replacement=True).item()
        out.append(itos[ix])
        if ix == 0:
            break
    print(''.join(out))
