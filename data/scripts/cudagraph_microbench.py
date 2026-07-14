"""Standalone feasibility + ceiling test for CUDA-graphing the prior decode.

Mimics the GPC prior's transformer (6 layers, d=1024, ff=4096, 8 heads,
8 tokens, vocab 59049) and compares, at batch 2:
  A) eager growing-sequence decode (current behavior: recompute prefix each token)
  B) CUDA-graph replay of a fixed-shape [B, max_seq, D] forward per token

No Isaac needed — pure torch, boots in seconds.
"""

import time
import torch
import torch.nn as nn

torch.manual_seed(0)
dev = "cuda"
B, D, FF, L, H, T, V = 2, 1024, 4096, 6, 8, 8, 59049
MAXLEN = T + 1


from torch.nn.attention import sdpa_kernel, SDPBackend

def build():
    layer = nn.TransformerEncoderLayer(
        d_model=D, nhead=H, dim_feedforward=FF, dropout=0.0,
        activation="gelu", batch_first=True,
    )
    # enable_nested_tensor=False disables the BetterTransformer fast path,
    # whose data-dependent branching breaks CUDA graph capture.
    enc = nn.TransformerEncoder(layer, num_layers=L, enable_nested_tensor=False).to(dev).eval()
    head = nn.Linear(D, V).to(dev).eval()
    tok_emb = nn.Embedding(V, D).to(dev).eval()
    pos = torch.randn(1, MAXLEN, D, device=dev)
    return enc, head, tok_emb, pos


enc, head, tok_emb, pos = build()
causal = torch.triu(torch.ones(MAXLEN, MAXLEN, device=dev) * float("-inf"), diagonal=1)


@torch.no_grad()
def eager_decode(context):
    """Current behavior: rebuild [ctx, t0..tk] and run full transformer each token."""
    generated = []
    for _ in range(T):
        if generated:
            toks = torch.stack(generated, dim=1)
            seq = torch.cat([context.unsqueeze(1), tok_emb(toks)], dim=1)
        else:
            seq = context.unsqueeze(1)
        k = seq.shape[1]
        seq = seq + pos[:, :k]
        hid = enc(seq, mask=causal[:k, :k], is_causal=True)
        logits = head(hid[:, -1])
        generated.append(torch.multinomial(logits.softmax(-1), 1).squeeze(-1))
    return torch.stack(generated, dim=1)


# ---- CUDA graph path: fixed [B, MAXLEN, D] buffer, static mask, replay ----
static_in = torch.zeros(B, MAXLEN, D, device=dev)
static_hidden = None
g = torch.cuda.CUDAGraph()

with torch.no_grad():
    # warmup on a side stream (required before capture)
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s), sdpa_kernel(SDPBackend.MATH):
        for _ in range(3):
            _h = enc(static_in, mask=causal, is_causal=True)
            _l = head(_h)
    torch.cuda.current_stream().wait_stream(s)

    with torch.cuda.graph(g), sdpa_kernel(SDPBackend.MATH):
        static_hidden = enc(static_in, mask=causal, is_causal=True)
        static_logits = head(static_hidden)


@torch.no_grad()
def graph_decode(context):
    """Fixed-shape padded buffer; replay the captured forward each token."""
    static_in.zero_()
    static_in[:, 0] = context + pos[:, 0]
    generated = []
    for step in range(T):
        g.replay()
        logits = static_logits[:, step]  # position `step` predicts token `step`
        nxt = torch.multinomial(logits.softmax(-1), 1).squeeze(-1)
        generated.append(nxt)
        if step + 1 < MAXLEN:
            static_in[:, step + 1] = tok_emb(nxt) + pos[:, step + 1]
    return torch.stack(generated, dim=1)


def timeit(fn, ctx, n=50):
    for _ in range(5):
        fn(ctx)
    torch.cuda.synchronize()
    t = time.perf_counter()
    for _ in range(n):
        fn(ctx)
    torch.cuda.synchronize()
    return 1000 * (time.perf_counter() - t) / n


ctx = torch.randn(B, D, device=dev)
try:
    e = timeit(eager_decode, ctx)
    gph = timeit(graph_decode, ctx)
    print(f"eager growing-seq decode : {e:6.2f} ms/decode")
    print(f"cuda-graph fixed-buf     : {gph:6.2f} ms/decode   ({e/gph:.1f}x)")
    print("CUDA GRAPH CAPTURE: OK")
except Exception as exc:
    import traceback
    traceback.print_exc()
    print("CUDA GRAPH CAPTURE: FAILED —", exc)
