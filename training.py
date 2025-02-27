from IPython.display import Image
import torch
import torch.nn as nn
from torch.nn import functional as F
import pickle
import argparse

parser = argparse.ArgumentParser(description="Testing")
parser.add_argument('-batch_size', type=str, required=True, help='Please provide a batch_size')
args = parser.parse_args()

# Check for cuda
print(f"batch size: {args.batch_size}")
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)

block_size = 128
batch_size = int(args.batch_size)
max_iters = 100
eval_iters = 100
learning_rate = 3e-4
n_embd = 384
n_layer = 4
n_head = 4
dropout = 0.2

# Read text dataset
with open("TextDataset.txt", "r", encoding="utf-8") as dataset:
    text = dataset.read()

# Tokeniser
chars = sorted(set(text)) # The types of characters that are present inside this text
# print(chars)
# print(len(chars)) # Number of integers we will get after encoding
vocab_size = len(chars)

# Encoder
# Convert string to integer
string_to_int = {ch:i for i,ch in enumerate(chars)} # This dictionary has key:value relationship as string:int (string -> int)
encoder = lambda e: [string_to_int[c] for c in e]

# Decoder
# Convert integer to string
int_to_string = {i:ch for i,ch in enumerate(chars)} # This dictionary has key:value relationship as int:string (int -> string)
decoder = lambda d: ''.join([int_to_string[c] for c in d])

# Simple example of how encoder and decoder works
# print(encoder("hello"))
# encoded = encoder("hello")
# decoded = decoder(encoded)
# print(decoded)

# Tensors
# Tensors are similar to arrays, but more optimised for GPUs, mainly used for ML/AI
data = torch.tensor(encoder(text), dtype=torch.long)

# Splitting into training and validation sets
# Usually, we do 80-20 splits (5-Means Clustering)
n = int(0.8 * len(data))
training_data = data[:n]
validation_data = data[n:]

def get_batch(split):
    data = training_data if split == 'train' else validation_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval() # dropout is turned off during evaluation
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            x, y = get_batch(split)
            logits, loss = model(x, y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train() # dropout is turned on during training
    return out

class Head(nn.Module):
    """ one head of self attention """
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # input of size (batch, time=step, channel)
        # output of size (batch, time=step, head_size)
        batch, time, channel = x.shape
        k = self.key(x) # (batch, time, head_size)
        q = self.query(x) # (batch, time, head_size)
        # compute attention scores ("affinities")
        """ Dot product step with scaling """
        # Transpose: flip second last (-2) dimension (time) with last (-1) dimension (head_size)
        wei = q @ k.transpose(-2,-1) * k.shape[-1] ** -0.5 # (batch, time, head_size) @ (batch, head_size, time) -> (batch, time, time)
        """ Masking step """
        # we expose the next token for every time step
        wei = wei.masked_fill(self.tril[:time, :time] == 0, float('-inf')) # (batch, time, time)
        """ Applying softmax """
        wei = F.softmax(wei, dim=-1) # (batch, time, time)
        wei = self.dropout(wei) # dropout to prevent overfitting
        # perform the weighted aggregation of the values
        v = self.value(x) # (batch, time, head_size)
        """ Matrix Multiplication with Value """
        out = wei @ v # (batch, time, time) @ (batch, time, head_size) => (batch, time, head_size)
        return out

class MultiHeadAttention(nn.Module):
    """ multiple heads of self-attention in parallel """
    def __init__(self, num_heads, head_size):
        super().__init__()
        # ModuleList runs the heads in parallel using the GPU (cuda), as oppposed to nn.Sequential which runs sequentially
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(head_size * num_heads, n_embd) # dot product, and add a bias term, we can use self.proj.bias to print out bias term
        self.dropout = nn.Dropout(dropout) # percentage chance that each value will be dropped

    def forward(self, x):
         # (batch, time, channel), dim=-1 => channel dimension => we are addng terms into the last dimension's vector
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out

class FeedForward(nn.Module):
    """ a simple linear layer followed by a non-linearity """
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout) # percentage of neurons will be turned to 0 to prevent overfitting
        )

    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    """ Transformer Block: Communication followed by computation """
    def __init__(self, n_embd, n_head):
        # n_embd: embedding dimension, n_head: the number of heads we'd like
        super().__init__()
        head_size = n_embd // n_head # no. of 'features' => the number of indices in our embedding vector
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedForward(n_embd) # Linear => ReLU => Linear
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        y = self.sa(x)
        x = self.ln1(x + y)
        y = self.ffwd(x)
        x = self.ln2(x + y)
        return x

class GPTLanguageModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__() # for inheritance purposes
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd) # Embedding
        self.position_embedding_table = nn.Embedding(block_size, n_embd) # Positional Embedding
        
        # no. of blocks corresponding to the no. of layers in the neral network
        # Sequential nn requires output from the previous block before running the current block => sequential computation, NOT parallel computation
        self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head) for _ in range(n_layer)]) 
        self.ln_f = nn.LayerNorm(n_embd) # final layer normalisation
        self.lm_head = nn.Linear(n_embd, vocab_size) # transformation

        self.apply(self._init_weights)
        
    # weight initialisation
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            
    def forward(self, index, targets=None):
        batch, time = index.shape
        
        # index and targets are both (batch, time) tensor of integers
        tok_emb = self.token_embedding_table(index) # (batch, time, channel)
        pos_emb = self.position_embedding_table(torch.arange(time, device=device)) # (time, channel)
        x = tok_emb + pos_emb # (batch, time, channel)
        x = self.blocks(x) # (batch, time, channel)
        x = self.ln_f(x) # (batch, time, channel)
        
        # This returns how likely this index occurs based on the embedding table
        logits = self.lm_head(x) # (batch, time, vocab_size)
        
        if targets is None:
            loss = None
        else:
            batch, time, channel = logits.shape
            logits = logits.view(batch * time, channel) # batch * time = N, channel = C (as seen from image below about cross_entropy)
            targets = targets.view(batch * time)
            loss = F.cross_entropy(logits, targets)
        return logits, loss

    # for the given index, we locate the next possible occurrences using the token_embedding table
    # then, we calculate its logits and loss with the pytorch functions (view and cross_entropy)
    # generate a probability distribution using the logits with the softmax function
    # the next index we generate based on the current index will be randomly selected based on this softmax function (probability rates DO play a factor in selection)
    def generate(self, index, max_new_tokens):
        # index is (batch, time) array of indices in the current context
        for _ in range(max_new_tokens):
            # crop idx to the last block_size tokens
            index_cond = index[:, -block_size:]
            # get the prediction
            logits, loss = self.forward(index_cond)
            # focus only on the last time step
            logits = logits[:, -1, :] # becomes (batch, channel)
            # apply softmax function to get probabilities (normalisation process, softmax is just one of many different normalisation techniques)
            probs = F.softmax(logits, dim=-1) # (batch, channel)
            # sample from the distribution
            index_next = torch.multinomial(probs, num_samples=1) # (batch, 1)
            # append sampled index to the running sequence
            index = torch.cat((index, index_next), dim=1) # (batch, time + 1)
        return index
    
model = GPTLanguageModel(vocab_size)
print("Loading model parameters...")
with open('GPTLanguageModel-1.pkl', 'rb') as f:
    model = pickle.load(f)
print('Loaded model parameters! :)')
m = model.to(device)

# PyTorch Optimiser
optimiser = torch.optim.AdamW(model.parameters(), lr=learning_rate)

# Training Process
for iter in range(max_iters):
    if iter % eval_iters == 0:
        losses = estimate_loss()
        print(f"step: {iter}, train losses: {losses['train']:.4f}, val losses: {losses['val']:.4f}")
    
    # sample a batch of data
    # inputs, targets -> xb, yb
    xb, yb = get_batch('train')

    # evaluate the loss
    logits, loss = model.forward(xb, yb)
    optimiser.zero_grad(set_to_none=True) # set to True to save memory, as 0 can take up space as an integer as well
    loss.backward() # this computes the new gradient with the differentiated function
    optimiser.step() # basically x2 = x1 - learning rate * new gradient
print(loss.item())

with open('GPTLanguageModel-1.pkl', 'wb') as f:
    pickle.dump(model, f)
print('Model parameters saved! :D')