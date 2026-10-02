import json, struct, math, collections
import torch
from tensorfold.cuda.exl3 import experts as x3experts
from tensorfold.families.glm5_next.cuda import exl3 as glmx3
import numpy as np

shard = "/model/model-00120-of-00120.safetensors"
f = open(shard, "rb")
n = struct.unpack("<Q", f.read(8))[0]
hdr = json.loads(f.read(n))
f.close()

per = collections.defaultdict(dict)
for k, v in hdr.items():
    if "trellis" not in k: continue
    parts = k.split(".")
    li, e, proj = parts[3], parts[6], parts[7]
    per[(li, e)][proj] = (k, v)

pure3 = next(k for k, v in sorted(per.items()) if all(hdr[v[p][0]]["shape"][-1] == 48 for p in ("gate_proj", "up_proj", "down_proj")))
pure4 = next(k for k, v in sorted(per.items()) if all(hdr[v[p][0]]["shape"][-1] == 64 for p in ("gate_proj", "up_proj", "down_proj")))
print("pure k3:", pure3, "| pure k4:", pure4, flush=True)

def pack(key):
    li, e = key
    outs = []
    for proj in ("gate_proj", "up_proj", "down_proj"):
        tkey, v = per[(li, e)][proj]
        a, b = v["data_offsets"]
        f = open(shard, "rb"); f.seek(a)
        t = torch.from_numpy(np.frombuffer(f.read(b - a), dtype="<i2").copy()).view(torch.int16).reshape(v["shape"]).cuda()
        f.close()
        suh_k = tkey.replace("trellis", "suh"); svh_k = tkey.replace("trellis", "svh")
        vs = hdr[suh_k]; a2, b2 = vs["data_offsets"]
        f = open(shard, "rb"); f.seek(a2)
        suh = torch.from_numpy(np.frombuffer(f.read(b2 - a2), dtype="<f2").copy()).view(torch.float16).cuda()
        vs2 = hdr[svh_k]; a3, b3 = vs2["data_offsets"]
        f = open(shard, "rb"); f.seek(a3)
        svh = torch.from_numpy(np.frombuffer(f.read(b3 - a3), dtype="<f2").copy()).view(torch.float16).cuda()
        f.close()
        outs.append((t, suh, svh))
    return tuple(outs)

pack3 = pack(pure3)
pack4 = pack(pure4)
ex = x3experts.prepare([pack3[0], pack4[0]], [pack3[1], pack4[1]], [pack3[2], pack4[2]], "mcg")

torch.manual_seed(0)
x = torch.randn(8, ex.dims, dtype=torch.float16, device="cuda")
pick = torch.zeros(8, 1, dtype=torch.int32, device="cuda")
pick[4:, 0] = 1

def ffn_cpu(xx, pk, bits):
    (tg, suhg, svhg), (tu, suhu, svhu), (td, suhd, svhd) = pk
    g = glmx3.forward(xx.double().cpu(), tg.cpu(), suhg.double().cpu(), svhg.double().cpu(), bits)
    u = glmx3.forward(xx.double().cpu(), tu.cpu(), suhu.double().cpu(), svhu.double().cpu(), bits)
    h = torch.nn.functional.silu(g) * u
    return glmx3.forward(h.double().cpu(), td.cpu(), suhd.double().cpu(), svhd.double().cpu(), bits).double().cpu()

s = x3experts.Scratch(ex, 8, 1)
y = x3experts.routed(x, pick, None, ex, s, None, 8, math.inf)
torch.cuda.synchronize()
y = y.float().cpu()
ref0 = ffn_cpu(x[:4].cpu(), pack3, 3)
ref1 = ffn_cpu(x[4:].cpu(), pack4, 4)
for rows, ref, name in ((slice(0, 4), ref0, "rows0-3 vs k3"), (slice(4, 8), ref1, "rows4-7 vs k4"), (slice(4, 8), ref0, "rows4-7 vs k3 (misroute check)")):
    d = (y[rows] - ref).abs()
    print(f"{name}: max {d.max().item():.4f} | mean {d.mean().item():.4f}", flush=True)
