"""Per-step CUDA graphs: capture one graph per decode length (1..T), replay
graph[k] at token step k. Matches eager's growing-sequence compute (no
padding waste) while removing per-launch CPU overhead."""
import time, torch, torch.nn as nn
from torch.nn.attention import sdpa_kernel, SDPBackend
torch.manual_seed(0); dev="cuda"
B,D,FF,L,H,T,V = 2,1024,4096,6,8,8,59049
MAXLEN=T+1
layer=nn.TransformerEncoderLayer(d_model=D,nhead=H,dim_feedforward=FF,dropout=0.0,activation="gelu",batch_first=True)
enc=nn.TransformerEncoder(layer,num_layers=L,enable_nested_tensor=False).to(dev).eval()
head=nn.Linear(D,V).to(dev).eval()
tok_emb=nn.Embedding(V,D).to(dev).eval()
pos=torch.randn(1,MAXLEN,D,device=dev)
causal=torch.triu(torch.ones(MAXLEN,MAXLEN,device=dev)*float("-inf"),diagonal=1)

@torch.no_grad()
def eager(context):
    gen=[]
    for _ in range(T):
        seq = context.unsqueeze(1) if not gen else torch.cat([context.unsqueeze(1),tok_emb(torch.stack(gen,1))],1)
        k=seq.shape[1]; seq=seq+pos[:,:k]
        hid=enc(seq,mask=causal[:k,:k],is_causal=True)
        gen.append(torch.multinomial(head(hid[:,-1]).softmax(-1),1).squeeze(-1))
    return torch.stack(gen,1)

# capture one graph per length k=1..T
graphs=[]; static_ins=[]; static_logits=[]
with torch.no_grad():
    for k in range(1,T+1):
        buf=torch.zeros(B,k,D,device=dev); static_ins.append(buf)
        s=torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s), sdpa_kernel(SDPBackend.MATH):
            for _ in range(3): _=head(enc(buf,mask=causal[:k,:k],is_causal=True)[:,-1])
        torch.cuda.current_stream().wait_stream(s)
        gk=torch.cuda.CUDAGraph()
        with torch.cuda.graph(gk), sdpa_kernel(SDPBackend.MATH):
            lg=head(enc(buf,mask=causal[:k,:k],is_causal=True)[:,-1])
        graphs.append(gk); static_logits.append(lg)

@torch.no_grad()
def graph_decode(context):
    gen=[]
    static_ins[0][:,0]=context+pos[:,0]
    for step in range(T):
        graphs[step].replay()
        nxt=torch.multinomial(static_logits[step].softmax(-1),1).squeeze(-1)
        gen.append(nxt)
        if step+1<T:
            # fill next graph's buffer: ctx + tokens 0..step
            nb=static_ins[step+1]
            nb[:,0]=context+pos[:,0]
            toks=torch.stack(gen,1)
            nb[:,1:step+2]=tok_emb(toks)+pos[:,1:step+2]
    return torch.stack(gen,1)

def timeit(fn,ctx,n=50):
    for _ in range(5): fn(ctx)
    torch.cuda.synchronize(); t=time.perf_counter()
    for _ in range(n): fn(ctx)
    torch.cuda.synchronize(); return 1000*(time.perf_counter()-t)/n

ctx=torch.randn(B,D,device=dev)
try:
    e=timeit(eager,ctx); gp=timeit(graph_decode,ctx)
    print(f"eager      : {e:6.2f} ms/decode")
    print(f"per-step gr: {gp:6.2f} ms/decode   ({e/gp:.1f}x)")
except Exception as ex:
    import traceback; traceback.print_exc(); print("FAILED",ex)
