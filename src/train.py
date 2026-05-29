import wandb
import os
import math
import time
import sys
import pickle
from contextlib import nullcontext
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
import tiktoken

W = 78
DOUBLE = "=" * W
SINGLE = "-" * W
TICK = "best"
ARROW = ">"

LOG_DIR = Path(__file__).resolve().parent / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)
LOG_PATH = LOG_DIR / f"run_{time.strftime('%Y%m%d_%H%M%S')}.txt"
SCRIPT_DIR = Path(__file__).resolve().parent


def log(message=""):
    line = "" if message == "" else f"{time.strftime('%Y-%m-%d %H:%M:%S')} | {message}"
    print(line)
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(f"{line}\n")


def header(title, subtitle=""):
    log()
    log(DOUBLE)
    log(f"  {title}")
    if subtitle:
        log(f"  {subtitle}")
    log(DOUBLE)


def row(label, value="", unit="", note=""):
    label_col = f"  {label:<28}"
    value_col = f"{str(value):<20}"
    unit_col = f"{unit:<8}"
    note_col = f"  {note}" if note else ""
    log(f"{label_col}{value_col}{unit_col}{note_col}")


def rule():   log(f"  {SINGLE}")
def blank():  log()
def info(msg):    log(f"  {ARROW}  {msg}")
def success(msg): log(f"  ok  {msg}")


log(f"{'llm':^{W}}")
blank()
start = time.time()

cleaned_path = Path(os.environ.get("data", SCRIPT_DIR / "input.txt"))
train_split = 0.9
seed = 1337

batch_size = 16
block_size = 32
max_iters = 6000
eval_interval = 100
eval_iters = 20
learning_rate = 1e-3
n_embd = 64
n_head = 4
n_layer = 4
dropout = 0.1
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0
decay_lr = True
warmup_iters = 100
lr_decay_iters = max_iters
min_lr = 1e-4
gradient_accumulation_steps = 1
dtype = 'bfloat16' if torch.cuda.is_available(
) and torch.cuda.is_bf16_supported() else 'float16'
compile_model = False
wandb_log = False
wandb_project = 'quadtrix'
wandb_run_name = 'run'

ddp = int(os.environ.get('RANK', -1)) != -1
if ddp:
    init_process_group(backend='nccl')
    ddp_rank = int(os.environ['RANK'])
    ddp_local_rank = int(os.environ['LOCAL_RANK'])
    ddp_world_size = int(os.environ['WORLD_SIZE'])
    device = f'cuda:{ddp_local_rank}'
    torch.cuda.set_device(device)
    master_process = ddp_rank == 0
    seed_offset = ddp_rank
    assert gradient_accumulation_steps % ddp_world_size == 0
    gradient_accumulation_steps //= ddp_world_size
else:
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    master_process = True
    seed_offset = 0
    ddp_world_size = 1

torch.manual_seed(seed + seed_offset)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

device_type = 'cuda' if 'cuda' in device else 'cpu'
ptdtype = {'float32': torch.float32,
           'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(
    device_type=device_type, dtype=ptdtype)
scaler = torch.cuda.amp.GradScaler(enabled=(dtype == 'float16'))


def get_tokenizer(encoding_name="o200k_base"):
    tokenizer = tiktoken.get_encoding(encoding_name)
    vocab_size = tokenizer.n_vocab
    return tokenizer, vocab_size


def encode(text, tokenizer): return tokenizer.encode(text)
def decode(tokens, tokenizer): return tokenizer.decode(tokens)


with open(cleaned_path, 'r', encoding='utf-8') as f:
    text = f.read()

tokenizer, vocab_size = get_tokenizer("o200k_base")
encoded_data = encode(text, tokenizer)

data = torch.tensor(encoded_data, dtype=torch.long)
n = int(train_split * len(data))
train_data = data[:n]
val_data = data[n:]


def get_batch(split):
    data_split = train_data if split == 'train' else val_data
    ix = torch.randint(len(data_split) - block_size, (batch_size,))
    x = torch.stack([data_split[i:i + block_size] for i in ix])
    y = torch.stack([data_split[i + 1:i + block_size + 1] for i in ix])
    if device_type == 'cuda':
        x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(
            device, non_blocking=True)
    else:
        x, y = x.to(device), y.to(device)
    return x, y


@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            with ctx:
                _, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out


def get_lr(it):
    if it < warmup_iters:
        return learning_rate * (it + 1) / (warmup_iters + 1)
    if it > lr_decay_iters:
        return min_lr
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (learning_rate - min_lr)


class QuadtrixHead(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(
            torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)
        wei = q @ k.transpose(-2, -1) * k.shape[-1] ** -0.5
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        return wei @ self.value(x)


class QuadtrixMHA(nn.Module):
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([QuadtrixHead(head_size)
                                   for _ in range(num_heads)])
        self.proj = nn.Linear(head_size * num_heads, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        return self.dropout(self.proj(out))


class QuadtrixFFN(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class QuadtrixBlock(nn.Module):
    def __init__(self, n_embd, n_head):
        super().__init__()
        head_size = n_embd // n_head
        self.sa = QuadtrixMHA(n_head, head_size)
        self.ffwd = QuadtrixFFN(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x


class Quadtrix(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(
            *[QuadtrixBlock(n_embd, n_head=n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        tok_emb = self.token_embedding_table(idx)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device))
        x = tok_emb + pos_emb
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)
        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)
            loss = F.cross_entropy(logits, targets)
        return logits, loss

    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx

    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        decay_params = [p for p in self.parameters(
        ) if p.requires_grad and p.dim() >= 2]
        no_decay_params = [
            p for p in self.parameters() if p.requires_grad and p.dim() < 2]
        optim_groups = [
            {'params': decay_params,    'weight_decay': weight_decay},
            {'params': no_decay_params, 'weight_decay': 0.0},
        ]
        use_fused = device_type == 'cuda' and 'fused' in torch.optim.AdamW.__init__.__code__.co_varnames
        return torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, fused=use_fused if use_fused else False)


model = Quadtrix().to(device)
n_params = sum(p.numel() for p in model.parameters())
optimizer = model.configure_optimizers(
    weight_decay, learning_rate, (beta1, beta2), device_type)


def count_activations(m, bs, seq_len, dev):
    total = 0
    hooks = []

    def _hook(module, inp, out):
        nonlocal total
        if isinstance(out, torch.Tensor):
            total += out.numel()
    for mod in m.modules():
        hooks.append(mod.register_forward_hook(_hook))
    dummy = torch.zeros(bs, seq_len, dtype=torch.long, device=dev)
    with torch.no_grad():
        m(dummy)
    for h in hooks:
        h.remove()
    return total


_num_activations = count_activations(model, batch_size, block_size, device)
_train_batches = len(train_data) // (batch_size * block_size)
_val_batches = len(val_data) // (batch_size * block_size)

if compile_model:
    model = torch.compile(model)

if ddp:
    model = DDP(model, device_ids=[ddp_local_rank])

raw_model = model.module if ddp else model

if wandb_log and master_process:

    wandb.init(project=wandb_project, name=wandb_run_name)

if master_process:
    row("Batch size", batch_size)
    row("Block size", block_size)
    row("Parameters", f"{n_params:,}")
    row("Grad accum steps", gradient_accumulation_steps)
    row("dtype", dtype)
    blank()

    header("TRAINING",
           f"{max_iters:,} steps | eval every {eval_interval} | checkpoint on improvement")
    blank()

best_val_loss = float('inf')
train_start = time.time()

for iter in range(max_iters):

    lr = get_lr(iter) if decay_lr else learning_rate
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    if iter % eval_interval == 0 or iter == max_iters - 1:
        if master_process:
            losses = estimate_loss()
            is_best = losses['val'] < best_val_loss
            if is_best:
                best_val_loss = losses['val']
                torch.save(raw_model.state_dict(), 'best_model.pt')
            log(f"  val loss {losses['val']:.6f}")
            if wandb_log:

                wandb.log(
                    {"iter": iter, "train/loss": losses['train'], "val/loss": losses['val'], "lr": lr})
            sys.stdout.flush()

    step_start = time.time()

    for micro_step in range(gradient_accumulation_steps):
        if ddp:
            model.require_backward_grad_sync = (
                micro_step == gradient_accumulation_steps - 1)
        xb, yb = get_batch('train')
        with ctx:
            logits, loss = model(xb, yb)
            loss = loss / gradient_accumulation_steps
        scaler.scale(loss).backward()

    if grad_clip != 0.0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

    grad_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            grad_norm += p.grad.detach().data.norm(2).item() ** 2
    grad_norm = grad_norm ** 0.5

    scaler.step(optimizer)
    scaler.update()
    optimizer.zero_grad(set_to_none=True)

    step_dt_ms = (time.time() - step_start) * 1000
    tok_per_sec = (batch_size * block_size *
                   gradient_accumulation_steps) / (step_dt_ms / 1000.0)
    cur_lr = optimizer.param_groups[0]['lr']

    if master_process:
        line = (f"step {iter} | loss: {loss.item() * gradient_accumulation_steps:.6f} | lr {cur_lr:.4e} | norm: {grad_norm:.4f} | dt: {step_dt_ms:.2f}ms | tok/sec: {tok_per_sec:.2f}")
        print(line)
        with open(LOG_PATH, "a", encoding="utf-8") as f:
            f.write(line + "\n")

    if master_process and (iter + 1) % 50 == 0:
        raw_model.eval()
        context = torch.zeros((1, 1), dtype=torch.long, device=device)
        with torch.no_grad():
            sample_ids = raw_model.generate(context, max_new_tokens=100)
        sample_text = decode(sample_ids[0].tolist(), tokenizer)
        raw_model.train()
        blank()
        log(f"  [sample @ step {iter + 1}]")
        log(f"  {'-' * 60}")
        log(f"  {sample_text.strip()}")
        log(f"  {'-' * 60}")
        blank()

if ddp:
    destroy_process_group()

if master_process:
    total_time = time.time() - train_start
    blank()
    rule()
    row("Duration",
        f"{int(total_time // 60)}m {int(total_time % 60):02d}s")
    row("Best val loss", f"{best_val_loss:.4f}", "", TICK)
    row("Checkpoint",    "best_model.pt",        "", TICK)
    rule()

    blank()
    raw_model.load_state_dict(torch.load(
        'best_model.pt', map_location=device, weights_only=True))
    raw_model.eval()
    success(f"Restored best_model.pt | val loss {best_val_loss:.4f}")

    header("INFERENCE", "quit / exit / q -> end session")
    blank()

    try:
        while True:
            prompt = input(f"  user  {ARROW} ").strip()
            log(f"  You  {ARROW} {prompt}")

            if prompt.lower() in ("quit", "exit", "q"):
                blank()
                success("Session ended.")
                break

            if not prompt:
                continue

            encoded_prompt = encode(prompt, tokenizer)
            context = torch.tensor(
                [encoded_prompt], dtype=torch.long, device=device)

            with torch.no_grad():
                output_ids = raw_model.generate(context, max_new_tokens=200)

            new_tokens = output_ids[0][len(encoded_prompt):].tolist()
            response = decode(new_tokens, tokenizer).strip()

            blank()
            log(f"  Model {ARROW} {response}")
            blank()

    except KeyboardInterrupt:
        blank()
        success("Interrupted.")

    end = time.time()
    wall_clock = end - start

    blank()
    rule()
    row("Training", f"{int(total_time // 60)}m {int(total_time % 60):02d}s")
    row("Total",
        f"{int(wall_clock // 60)}m {int(wall_clock % 60):02d}s", "", TICK)
    rule()
    blank()
    log(f"{DOUBLE}\n")
