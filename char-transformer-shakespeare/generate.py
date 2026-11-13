import torch
import torch.nn as nn
from torch.nn import functional as F

# --- COPY ALL YOUR HYPERPARAMETERS EXACTLY ---
batch_size = 32
block_size = 128
max_iters = 5000
eval_interval = 100
learning_rate = 1e-3
device = 'mps' if torch.backends.mps.is_available() else 'cpu' # safe check
n_embd = 192
n_head = 6
n_layer = 6
dropout = 0.2
# ---------------------------------------------

# --- WE STILL NEED THE VOCABULARY TO DECODE ---
# (In a real app, you would save 'stoi' and 'itos' to a file too, 
# but for now, re-reading the text is the easiest way to rebuild them)
with open('shakespeare.txt', 'r', encoding='utf-8') as f:
    text = f.read()

chars = sorted(list(set(text)))
vocab_size = len(chars)
stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join([itos[i] for i in l])

# --- PASTE YOUR MODEL CLASSES HERE ---
# (Head, MultipleHeadAttention, feedForward, Block, BigramLanguageModel)

class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)
    
    def forward(self,x):
        B,T,C = x.shape
        k = self.key(x)
        q = self.query(x)
        wei = q @ k.transpose(-2,-1) * k.shape[-1]**-0.5
        wei = wei.masked_fill(self.tril[:T,:T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        v = self.value(x)
        out = wei @ v
        return out

class MultipleHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(head_size*num_heads, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self,x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out

class feedForward(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4*n_embd),
            nn.ReLU(),
            nn.Linear(4*n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    def __init__(self, n_embd, n_head):
        super().__init__()
        head_size = n_embd//n_head
        self.sa = MultipleHeadAttention(n_head, head_size)
        self.ffwd = feedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self,x):
        x = x+self.sa(self.ln1(x))
        x = x+self.ffwd(self.ln2(x))
        return x

class BigramLanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        # each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd,n_head=n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):
        B,T = idx.shape 

        # idx and targets are both (B,T) tensor of integers
        tok_emb = self.token_embedding_table(idx) # (B,T,C)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device))
        x = tok_emb + pos_emb
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)

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
            idx_cond = idx[:,-block_size:]
            logits, loss = self(idx_cond)
            logits = logits[:, -1, :] 
            probs = F.softmax(logits, dim=-1) 
            idx_next = torch.multinomial(probs, num_samples=1) 
            idx = torch.cat((idx, idx_next), dim=1) 
        return idx
        
# ----------------------------------------------------
# --- THIS IS THE NEW PART: LOAD INSTEAD OF TRAIN ---
# ----------------------------------------------------

print("Initializing model architecture...")
model = BigramLanguageModel()
m = model.to(device)

print(f"Loading model weights from 'model.pth' to {device}...")
# map_location ensures it loads onto the correct device (CPU or MPS)
m.load_state_dict(torch.load('model.pth', map_location=device))

# Switch to evaluation mode (important for Dropout/LayerNorm)
m.eval()

print("Model loaded! Generating text...")
print("-" * 50)

# Generate
context = torch.zeros((1, 1), dtype=torch.long, device=device)
output = m.generate(context, max_new_tokens=1000)
print(decode(output[0].tolist()))

# ... (Previous code: Imports, Model Classes, Loading Weights) ...

print("\n" + "="*40)
print("     Talk to Shakespeare GPT")
print("="*40)

while True:
    # 1. Get input from user
    start_str = input("\n\nPrompt (type 'exit' to quit): ")
    
    if start_str.lower() == 'exit':
        break
    
    # 2. Encode the string to integers
    # IMPORTANT: The model only knows characters from shakespeare.txt. 
    # If you type a character it hasn't seen (like '€' or an emoji), this will crash.
    try:
        context_idx = encode(start_str)
    except KeyError:
        print("⚠️ Error: You used a character the model doesn't know. Try standard letters/punctuation.")
        continue

    # 3. Convert to tensor and move to GPU/MPS
    # unsqueeze(0) adds the batch dimension: (T) -> (1, T)
    context = torch.tensor(context_idx, dtype=torch.long, device=device).unsqueeze(0)

    # 4. Generate
    print("Generating...", end="", flush=True)
    # We generate 200 tokens. You can change this number.
    output = m.generate(context, max_new_tokens=200)
    
    # 5. Decode and print
    # The output contains [Your Prompt + Generated Text]
    print("\n" + "-"*20)
    print(decode(output[0].tolist()))
    print("-"*20)