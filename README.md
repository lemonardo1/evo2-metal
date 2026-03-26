# evo2-metal

Run [ARC Institute's Evo 2](https://github.com/arcinstitute/evo2) DNA language model on **Apple Silicon (Metal GPU)** — no CUDA required.

Evo 2 is a 7B–40B parameter genomic foundation model trained on 9.3 trillion DNA tokens. This package makes the 7B models fully usable on a Mac with M-series chips.

## How it works

Evo 2 depends on `flash-attn`, which requires NVIDIA CUDA and does not run on Mac. This package provides:

- A **FlashAttention v2 kernel written in Metal Shading Language** (Apple's GPU compute API), dispatched via PyObjC.
- A **set of monkey-patches** applied before importing `evo2` that redirect all CUDA-dependent calls to CPU/Metal equivalents — without modifying the original Evo 2 source code.

```
import evo2_metal   ←  patches flash_attn, vortex, and evo2.scoring
from evo2 import Evo2
```

## Installation

**Requires:** macOS with Apple Silicon (M1/M2/M3/M4/M5), Python 3.11 or 3.12.

```bash
# 1. Create a Python 3.12 environment (evo2 requires <3.13)
conda create -n evo2-mac python=3.12
conda activate evo2-mac

# 2. Install PyTorch (CPU build for Mac)
pip install torch --index-url https://download.pytorch.org/whl/cpu

# 3. Install evo2-metal (PyPI)
pip install evo2-metal

# Alternative: install directly from GitHub
# pip install git+https://github.com/lemonardo1/evo2-metal.git
```

## Usage

### Sequence scoring

```python
import evo2_metal          # must be first
from evo2 import Evo2

model = Evo2('evo2_7b_base')

scores = model.score_sequences([
    "ATGAAAGCAATTTTCGTACTGAAAGGTTCAGGT",   # wildtype
    "ATGAAAGCAATTTTCGTACTGAAAGGTTCAGGA",   # variant
])
print(scores)  # [np.float32(-1.27), np.float32(-1.43)]
```

### DNA sequence generation

```python
import evo2_metal
from evo2 import Evo2

model = Evo2('evo2_7b_base')

output = model.generate(
    prompt_seqs=["ATGAAAGCAATTTTCGTACTGAAAGGTTCAGGT"],
    n_tokens=200,
    temperature=1.0,
    top_k=4,
)
print(output.sequences[0])
```

## Supported models

| Model | Parameters | Context | Supported |
|-------|-----------|---------|-----------|
| `evo2_7b_base` | 7B | 8K | ✅ |
| `evo2_7b` | 7B | 1M | ✅ |
| `evo2_7b_262k` | 7B | 262K | ✅ |
| `evo2_7b_microviridae` | 7B | 8K | ✅ |
| `evo2_1b_base` | 1B | 8K | ❌ (requires FP8/Transformer Engine) |
| `evo2_40b` / `evo2_20b` | 40B / 20B | 1M | ❌ (requires multi-GPU) |

## Benchmarks

Measured on **Apple M5 Max (128 GB)**, PyTorch 2.10, macOS 15.

### Evo 2 7B — actual architecture specs (d=128, H=32, causal, Metal GPU)

Evo 2 7B uses StripedHyena 2 (Hyena conv + MHA interleaved). The MHA layers use:
`d_model=4096`, `num_heads=32`, `d_head=128`, RoPE positional encoding.

| Context length | Time (ms) | TFLOP/s |
|---|---|---|
| N=256 | 6.56 ms | 0.164 |
| N=512 | 9.36 ms | 0.459 |
| N=1024 | 22.55 ms | 0.762 |
| N=2048 | 71.92 ms | 0.955 |
| N=4096 | 265.62 ms | 1.035 |
| N=8192 | 1007.82 ms | 1.091 |

Accuracy (d=128, H=32, causal): max error < 2e-6 (float32 precision), all PASS.

### FlashAttention v2 kernel: float4 vectorization speedup

| Sequence length | Head dim | v1 (scalar) | v2 (float4) | Speedup |
|---|---|---|---|---|
| N=512 | d=64 | 3.91 ms | 1.31 ms | 2.99x |
| N=1024 | d=64 | 7.28 ms | 2.40 ms | 3.03x |
| N=2048 | d=64 | 14.44 ms | 4.59 ms | 3.14x |
| N=1024 | d=128 | 14.27 ms | 4.46 ms | 3.20x |

> **Note:** Only the FlashAttention kernel runs on Metal GPU. All other operations (FFN, embeddings, sampling) run on CPU. Overall generation speed depends on model size and sequence length.

To reproduce:
```bash
python -m evo2_metal.flash_attention_metal
```

## Patches applied

| Patch | Reason |
|-------|--------|
| `torch.cuda.device` → no-op for cpu | `StripedHyena.__init__` calls `torch.cuda.device("cpu")` |
| `torch.cuda.memory_allocated` → returns 0 | Called in generation loop for logging |
| `flash_attn_2_cuda` → mocked | Not available on Mac |
| `FlashSelfAttention.forward` → Metal backend | `assert qkv.is_cuda` fails on CPU |
| `FlashCrossAttention.forward` → Metal backend | Same |
| `local_flash_attn_with_kvcache` → PyTorch SDPA | KV-cache path during token generation |
| `vortex.generation.generate` → `device='cpu'` | Default was `'cuda:0'` |
| `evo2.scoring.prepare_batch` → `device='cpu'` | Default was `'cuda:0'` |

## Verified on

- Apple M5 Max, macOS 15 (Sequoia)
- Python 3.12, PyTorch 2.11, evo2 0.5.3

## License

Apache 2.0 — see [LICENSE](LICENSE).

Evo 2 model weights are released under their own license by ARC Institute. See the [Evo 2 repository](https://github.com/arcinstitute/evo2) for details.


## Contributor

github.com/lemonardo1
(Daeseong Kim)
lemonaatree@gmail.com
daeseongkim@yuhs.ac