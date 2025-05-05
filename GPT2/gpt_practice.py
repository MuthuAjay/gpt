import inspect
import math
import os
import torch
import torch.nn as nn
from torch.nn import functional as F
import requests
from dataclasses import dataclass
from torch.nn.parallel import DistributedDataParallel as DDP
import time
import tiktoken


# hyperparameters
batch_size = 64
block_size = 256
max_iters = 1000
eval_interval = 100
learning_rate = 1e-3
device = "cuda" if torch.cuda.is_available() else "cpu"
eval_iters = 200
n_embd = 384
n_head = 6
n_layer = 6
dropout = 0.2


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, config.n_embd * 4)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(config.n_embd * 4, config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1

    def forward(self, x: torch.Tensor):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x


class Block(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CasualSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class CasualSelfAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embd, config.n_embd * 3)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = (
            1  # this is because of the increasing variance due to residual connections
        )

        # regularization
        self.n_head = config.n_head
        self.n_embd = config.n_embd

        # not really  a bias, more of a mask, but following the OpenAI/Hugging Face naming though
        self.register_buffer(
            "bias",
            torch.tril(torch.ones(config.block_size, config.block_size)).view(
                1, 1, config.block_size, config.block_size
            ),
        )

    def forward(self, x: torch.Tensor):
        B, T, C = x.size()  # batch, sequence length, embedding dimensionality (n_embd)
        # calculate query, key, values for all heads in batch and move head forward to be the batch dimension
        # nh is 'number of heads' , hs is 'head size' and C is 'embedding dimension'
        # e.g in GPT-2 (124M), n_head = 12, hs=64, so nh*hs =C=768 channels in the transformer
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(
            1, 2
        )  # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(
            1, 2
        )  # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(
            1, 2
        )  # (B, nh, T, hs)

        # attention (materialize the large (T,T) matrix for all the queries and keys)
        # att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        # att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float("-inf"))
        # att = F.softmax(att, dim=-1)
        # y = att @ v  # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)

        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)

        y = (
            y.transpose(1, 2).contiguous().view(B, T, C)
        )  # re-assemble all head outputs side by side

        # output projection
        y = self.c_proj(y)
        return y


@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = (
        50257  # GPT2 vocab size with 50256 BPE Merges and EOT token (End of Text)
    )
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 756
    bias: bool = True
    dropout: float = 0.0


class DataLoaderLite:

    def __init__(self, B, T):
        self.B = B
        self.T = T
        text = self.download_data()
        enc = tiktoken.get_encoding("gpt2")
        self.enc = tiktoken.get_encoding("gpt2")
        tokens = enc.encode(text)
        self.tokens = torch.tensor(tokens, dtype=torch.long)
        print(f"loaded text of length {len(self.tokens)} tokens")
        print(f"1 epoch = {len(self.tokens) // (B * T)} batches")
        # state
        self.current_position = 0

    def download_data(self):
        torch.manual_seed(1337)
        if not os.path.exists("input.txt"):
            req = requests.get(
                r"https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt",
                verify=False,
            )
            with open("input.txt", "wb") as f:
                f.write(req.content)
        with open("input.txt", "r", encoding="utf-8") as f:
            text = f.read()
        return text

    def next_batch(self):
        B, T = self.B, self.T
        buf = self.tokens[self.current_position : self.current_position + B * T + 1]
        x = (buf[:-1].view(B, T)).to(device)  # inputs
        y = (buf[1:].view(B, T)).to(device)  # targets

        # advance the position in the tensor
        self.current_position += B * T
        # if loading the next batch would overrun the data, reset the position
        if self.current_position + (B * T + 1) > len(self.tokens):
            self.current_position = 0

        return x, y


class GPT(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()

        assert (
            config.n_layer % config.n_head == 0
        ), "n_layer must be divisible by n_head"
        assert config.vocab_size is not None, "vocab_size must be specified"
        assert config.block_size is not None, "block_size must be specified"

        self.config = config

        self.transformer = nn.ModuleDict(
            dict(
                wte=nn.Embedding(config.vocab_size, config.n_embd),  # word embeddings
                wpe=nn.Embedding(
                    config.block_size, config.n_embd
                ),  # position embeddings
                h=nn.ModuleList(
                    [Block(config) for _ in range(config.n_layer)]
                ),  # transformer blocks
                ln_f=nn.LayerNorm(config.n_embd),  # final layer norm
            )
        )

        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # weight sharing scheme
        self.transformer.wte.weight = self.lm_head.weight

        # self.init_weights()
        self.apply(self._init_weights)

    def _init_weights(self, module):

        if isinstance(module, nn.Linear):
            std = 0.02
            if hasattr(module, "NANOGPT_SCALE_INIT"):
                std *= (
                    2 * self.config.n_layer
                ) ** -0.5  # here 2 is because of the 2 residual connections from the MLP and the attention
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)

            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx: torch.Tensor, targets: torch.Tensor = None):

        # idx is the input tensor of shap (B, T) (batch size, sequence length or block size)
        B, T = idx.size()
        assert (
            T <= self.config.block_size
        ), f"Cannot forward, sequence of length {T} is longer than block size {self.config.block_size}"

        # forward the token and the position embeddings
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)
        pos_emb = self.transformer.wpe(pos)  # position embeddings of shape (T, n_embd)
        tok_emb = self.transformer.wte(idx)  # token embeddings of shape (B, T, n_embd)
        x = tok_emb + pos_emb  # (B, T, n_embd)

        # forward the transformer blocks
        for block in self.transformer.h:
            x = block(x)

        # apply the final layer norm
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)  # (B, T, vocab_size)

        loss = None

        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),  # (B*T, vocab_size)
                targets.view(-1),  # (B*T, )
            )  # flatten the logits from (B, T, vocab_size) to (B*T, vocab_size)
        # return the logits and the loss
        return logits, loss

    def configure_optimizers(
        self, learning_rate: float = 1e-3, weight_decay: float = 0.1, device="cuda"
    ):

        # start with all the candidates parameters that require gradients

        param_dict = {pn: p for pn, p in self.named_parameters() if p.requires_grad}

        # create optim groups any parameter that is 2d will be weight decayed, or else No
        # i.e all the weight tensors in matmuls + embeddings will be weight decayed and all the biases and layernorm will not

        decay_params = [p for n, p in param_dict.items() if len(p.shape) >= 2]
        nodecay_params = [p for n, p in param_dict.items() if len(p.shape) < 2]

        # optimizer groups
        optim_groups = [
            {"params": decay_params, "weight_decay": weight_decay},
            {"params": nodecay_params, "weight_decay": 0.0},
        ]

        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(
            f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params} parameters"
        )
        print(
            f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params} parameters"
        )

        # create the AdamW optimizer and use the fused version if available
        fused_available = "fused" in inspect.signature(torch.optim.AdamW).parameters
        use_fuse = fused_available and "cuda" in device
        print(f"using fused AdamW: {use_fuse}")

        optimizer = torch.optim.AdamW(
            optim_groups,
            lr=learning_rate,
            betas=(0.9, 0.95),
            eps=1e-8,
            weight_decay=weight_decay,
            fused=use_fuse,
        )

        return optimizer


device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"using device: {device}")

torch.manual_seed(1337)
torch.cuda.manual_seed(1337)


# get a data loader
train_loader = DataLoaderLite(4, 32)


torch.set_float32_matmul_precision("high")

num_return_sequences = 5
max_length = 100

# load the model

model = GPT(GPTConfig()).to(device)
model = torch.compile(model)
model.train()

max_lr = 6e-4
min_lr = max_lr / 10
warmup_steps = 50
max_steps = 50


def get_lr(it):

    # 1) linear warmup for warmyp_iter steps
    if it < warmup_steps:
        return max_lr * (it + 1) / warmup_steps
    # 2) if it > lr_decay_iters ,return min_lr
    if it > max_steps:
        return min_lr

    # 3) in between , do the cosine decay down to min_lr
    decay_ratio = (it - warmup_steps) / (max_steps - warmup_steps)
    assert 0 <= decay_ratio <= 1

    coef = 0.5 * (1 + math.cos(math.pi * decay_ratio))  # coef starts at 1 and goes to 0

    return min_lr + (max_lr - min_lr) * coef


# create the optimizer
optimizer = model.configure_optimizers(
    learning_rate=6e-4, weight_decay=0.1, device=device
)

for step in range(max_steps):

    t0 = time.time()
    x, y = train_loader.next_batch()
    optimizer.zero_grad()
    # with torch.autocast(device_type=device, dtype=torch.bfloat16): # use this only in A100 GPUs
    #     logits, loss = model(x, y)
    #     loss.backward()
    logits, loss = model(x, y)
    loss.backward()

    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # gradient clipping
    lr = get_lr(step)

    for param_group in optimizer.param_groups:
        param_group["lr"] = lr
    optimizer.step()

    t1 = time.time()
    dt = (t1 - t0) * 1000  # in ms
    tokens_per_sec = (train_loader.B * train_loader.T) / (t1 - t0)

    print(
        f" step {step} | loss: {loss.item()} | lr {lr} | norm:{norm} |dt: {dt:.2f}s | tokens/sec: {tokens_per_sec:.2f}"
    )


import sys

# sys.exit(0)

# prefix tokens

# tokens = enc.encode("Hello, I'm a language model")
# tokens = torch.tensor(tokens, dtype=torch.long)  # (8,)
# tokens = tokens.unsqueeze(0).repeat(num_return_sequences, 1).to(device)  # (5, 8)
# x = tokens.to(device)

# generate ! right now x is (B, T) where B = 5 and T = 8
torch.manual_seed(42)
torch.cuda.manual_seed(42)
while x.size(1) < max_length:
    with torch.no_grad():
        logits, loss = model(x)  # (B, T, vocab_size)
        # take the logits at the last position
        logits = logits[:, -1, :]  # (B, vocab_size)
        # get the probabilities
        probs = F.softmax(logits, dim=-1)
        # do the top k sampling of 50 (hugging face uses 50 - default)
        # topk_probs here becomes (5, 50), topk_indices is (5, 50)
        topk_probs, topk_indices = torch.topk(probs, 50, dim=-1)
        # sample from the top k
        ix = torch.multinomial(topk_probs, 1)
        # gather the corredsponding indices
        xcol = torch.gather(topk_indices, -1, ix)
        # append to the sequence
        x = torch.cat((x, xcol), dim=1)

# print the generated text

for i in range(num_return_sequences):
    generated = train_loader.enc.decode(x[i, :max_length].tolist())
    print(generated)
    print("=" * 80)