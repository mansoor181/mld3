"""Strip a PairFlow/Lightning molecule MDLM checkpoint down to a plain tensor state_dict.

The SDTT harness loads its teacher with torch.load(..., weights_only=True), which refuses
the omegaconf objects Lightning stores alongside the weights. It then strips one leading
"backbone." from every key. Our checkpoints already carry exactly that prefix, so the only
thing needed is to drop everything that is not a tensor.
"""
import sys, types, torch

class _AnyClass:
    def __init__(self, *a, **k): pass
    def __setstate__(self, s): pass

def _install(name):
    m = types.ModuleType(name); m.__path__ = []
    m.__getattr__ = lambda n: _AnyClass
    sys.modules[name] = m
    return m

class _Finder:
    def find_module(self, fullname, path=None):
        return self if fullname.startswith("transformers_modules") else None
    def load_module(self, fullname):
        return sys.modules.get(fullname) or _install(fullname)

_install("transformers_modules")
sys.meta_path.insert(0, _Finder())

src, dst = sys.argv[1], sys.argv[2]
ck = torch.load(src, map_location="cpu")
sd = ck["state_dict"]
clean = {k: v.clone() for k, v in sd.items() if torch.is_tensor(v)}
torch.save({"state_dict": clean}, dst)

back = torch.load(dst, map_location="cpu", weights_only=True)["state_dict"]
assert len(back) == len(clean), "round trip lost tensors"
stripped = {k.replace("backbone.", "", 1) for k in back}
print("wrote %s" % dst)
print("  tensors: %d, global_step: %s" % (len(clean), ck.get("global_step")))
print("  vocab_embed: %s" % (tuple(back["backbone.vocab_embed.embedding"].shape),))
print("  sample stripped keys: %s" % sorted(stripped)[:3])
