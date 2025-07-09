import pandas as pd
import torch
import torch.nn as nn
from torch.nn import functional as F

block_size = 8
batch_size = 32
max_iters = 3000
eval_interval = 300
torch.manual_seed(1337)
learning_rate = 1e-3
eval_iters = 200
device = 'cuda' if torch.cuda.is_available() else 'cpu'
n_embd = 32
torch.manual_seed(1337)

with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()
    
chars = sorted(list(set(text)))
vocab_size = len(chars)

string_to_integer = {}
integer_to_string = {}
for i,k in enumerate(chars):
    string_to_integer[k] = i
    integer_to_string[i] = k
    
def encode(text):
    return [string_to_integer[i] for i in text]
def decode(text):
    return "".join([integer_to_string[i] for i in text])

data = torch.tensor(encode(text),dtype=torch.long)

train_data = data[:int(0.9 * len(data))]
val_data = data[int(0.9 * len(data)):]

x = train_data[:block_size]
y = train_data[1:block_size+1]
    
def get_batch(split):
    data = train_data if split == "train" else val_data
    ix = torch.randint(len(data) - block_size , (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x,y = x.to(device),y.to(device)
    return x,y

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

class BigramLanguageModel(nn.Module):

    def __init__(self):
        super().__init__()
        
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.lm_head = nn.Linear(n_embd,vocab_size)

    def forward(self, idx, targets=None):
        tok_emd = self.token_embedding_table(idx) # (B,T,C)
        logits = self.lm_head(tok_emd)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            logits, loss = self(idx)
            logits = logits[:, -1, :] # becomes (B, C)
            probs = F.softmax(logits, dim=-1) # (B, C)
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
        return idx

model = BigramLanguageModel()
m=model.to(device)

optimizer = torch.optim.AdamW(model.parameters() , lr=learning_rate)

for iter in range(max_iters):

    # every once in a while evaluate the loss on train and val sets
    if iter % eval_interval == 0:
        losses = estimate_loss()
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

    # sample a batch of data
    xb, yb = get_batch('train')

    # evaluate the loss
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

context = torch.zeros((1, 1), dtype=torch.long, device=device)
print(decode(m.generate(context, max_new_tokens=500)[0].tolist()))