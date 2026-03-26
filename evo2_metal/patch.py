"""
evo2-metal patch module.

Applies all necessary monkey-patches to make Evo 2 run on Apple Silicon
without CUDA. Must be imported before any evo2 / vortex modules.

Patches applied
---------------
1. torch.cuda.device           → no-op context manager for cpu/mps devices
2. torch.cuda.memory_allocated → returns 0 for non-CUDA devices
3. torch.cuda.empty_cache      → no-op (safe; only called in low_mem_mode)
4. torch.autocast("cuda")      → silently redirected to cpu autocast
5. flash_attn_2_cuda / triton  → mocked (not available on Mac)
6. vortex.FlashSelfAttention.forward  → Metal-backed (removes is_cuda assert)
7. vortex.FlashCrossAttention.forward → Metal-backed
8. local_flash_attn_with_kvcache      → PyTorch SDPA fallback for generation
9. vortex.model.generation.generate   → defaults device='cpu'
10. evo2.scoring.prepare_batch        → defaults device='cpu'
"""

import math
import sys
import torch
from unittest.mock import MagicMock
import numpy as np

_patched = False


def apply_patches():
    global _patched
    if _patched:
        return
    _patched = True

    _patch_cuda_device()
    _patch_cuda_memory()
    _patch_cuda_misc()
    _patch_autocast()
    _mock_flash_attn()
    _patch_vortex_attention()
    _patch_vortex_generation()
    _patch_evo2_scoring()


# ── 1. torch.cuda.device → no-op for cpu/mps ─────────────────────────────────

def _patch_cuda_device():
    _original = torch.cuda.device

    class _CudaDeviceCompat:
        """Drop-in for torch.cuda.device; silently no-ops on cpu/mps devices."""
        def __init__(self, device):
            device_str = str(device)
            self._is_noop = 'cuda' not in device_str
            if not self._is_noop:
                self._ctx = _original(device)

        def __enter__(self):
            if self._is_noop:
                return self
            return self._ctx.__enter__()

        def __exit__(self, *args):
            if not self._is_noop:
                return self._ctx.__exit__(*args)

    torch.cuda.device = _CudaDeviceCompat


# ── 2. torch.cuda.memory_allocated → 0 for non-CUDA ─────────────────────────

def _patch_cuda_memory():
    _original = torch.cuda.memory_allocated

    def _safe(device=None):
        try:
            return _original(device)
        except Exception:
            return 0

    torch.cuda.memory_allocated = _safe


# ── 3. torch.cuda.empty_cache → no-op; other misc cuda stubs ─────────────────

def _patch_cuda_misc():
    # empty_cache is called in vortex low_mem_mode; silently ignore on CPU
    torch.cuda.empty_cache = lambda: None


# ── 4. torch.autocast("cuda") → redirect to cpu ───────────────────────────────

def _patch_autocast():
    _original_autocast = torch.autocast

    class _AutocastCompat:
        """Redirect torch.autocast('cuda') → cpu autocast on non-CUDA systems."""
        def __init__(self, device_type, *args, **kwargs):
            if device_type == "cuda" and not torch.cuda.is_available():
                device_type = "cpu"
                kwargs.pop("dtype", None)   # cpu autocast defaults are fine
            self._ctx = _original_autocast(device_type, *args, **kwargs)

        def __enter__(self):
            return self._ctx.__enter__()

        def __exit__(self, *args):
            return self._ctx.__exit__(*args)

        def __call__(self, func):
            return self._ctx(func)

    torch.autocast = _AutocastCompat


# ── 5. Mock flash_attn_2_cuda / triton / flash_attn ──────────────────────────

def _mock_flash_attn():
    sys.modules['flash_attn_2_cuda'] = MagicMock()
    sys.modules['triton'] = MagicMock()
    sys.modules['triton.language'] = MagicMock()

    mock = MagicMock()
    mock.flash_attn_interface = MagicMock()
    sys.modules['flash_attn'] = mock
    sys.modules['flash_attn.flash_attn_interface'] = mock.flash_attn_interface

    # Rotary embedding: PyTorch fallback
    def apply_rotary_torch(x, cos, sin, seqlen_offsets=0, cu_seqlens=None,
                           max_seqlen=None, interleaved=False, inplace=False,
                           conjugate=False):
        ro_dim = cos.shape[-1] * 2
        x_ro, x_pass = x[..., :ro_dim], x[..., ro_dim:]
        if not interleaved:
            x0, x1 = x_ro[..., :ro_dim // 2], x_ro[..., ro_dim // 2:]
        else:
            x0, x1 = x_ro[..., ::2], x_ro[..., 1::2]
        seq_len = x.shape[1] if cu_seqlens is None else max_seqlen
        cos_s = cos[seqlen_offsets:seqlen_offsets + seq_len].unsqueeze(0).unsqueeze(2)
        sin_s = sin[seqlen_offsets:seqlen_offsets + seq_len].unsqueeze(0).unsqueeze(2)
        if conjugate:
            sin_s = -sin_s
        o0 = x0 * cos_s - x1 * sin_s
        o1 = x0 * sin_s + x1 * cos_s
        if not interleaved:
            res_ro = torch.cat([o0, o1], dim=-1)
        else:
            res_ro = torch.empty_like(x_ro)
            res_ro[..., ::2] = o0
            res_ro[..., 1::2] = o1
        out = torch.cat([res_ro, x_pass], dim=-1)
        if inplace:
            x.copy_(out)
            return x
        return out

    rot_mod = MagicMock()
    rot_mod.apply_rotary = apply_rotary_torch
    sys.modules['vortex.ops.embedding.rotary'] = rot_mod

    # Store for use in later patches
    apply_patches._apply_rotary_torch = apply_rotary_torch

    # Metal attention backend
    from evo2_metal.flash_attention_metal import MetalFlashAttention
    metal_fa = MetalFlashAttention(Br=32, Bc=32)

    def metal_flash_attn_func(*args, **kwargs):
        q = args[0] if len(args) >= 1 else kwargs['q']
        k = args[1] if len(args) >= 2 else kwargs['k']
        v = args[2] if len(args) >= 3 else kwargs['v']
        causal = kwargs.get('causal', False)
        orig_dtype = q.dtype
        q_np = q.detach().float().cpu().numpy()
        k_np = k.detach().float().cpu().numpy()
        v_np = v.detach().float().cpu().numpy()
        needs_transpose = q_np.ndim == 4
        if needs_transpose:
            q_np = np.transpose(q_np, (0, 2, 1, 3))
            k_np = np.transpose(k_np, (0, 2, 1, 3))
            v_np = np.transpose(v_np, (0, 2, 1, 3))
        out_np = metal_fa.forward(q_np, k_np, v_np, causal=causal)
        if needs_transpose:
            out_np = np.transpose(out_np, (0, 2, 1, 3))
        return torch.from_numpy(out_np).to(q.device, dtype=orig_dtype)

    apply_patches._metal_flash_attn_func = metal_flash_attn_func

    mock.flash_attn_func = metal_flash_attn_func
    mock.flash_attn_varlen_func = metal_flash_attn_func
    sys.modules['flash_attn'].flash_attn_func = metal_flash_attn_func
    sys.modules['flash_attn.flash_attn_interface'].flash_attn_func = metal_flash_attn_func


# ── 4-6. Patch vortex attention classes ──────────────────────────────────────

def _patch_vortex_attention():
    metal_flash_attn_func = apply_patches._metal_flash_attn_func
    apply_rotary_torch = apply_patches._apply_rotary_torch

    def _flash_self_attn_forward(self, qkv, causal=None, cu_seqlens=None, max_seqlen=None):
        causal_val = self.causal if causal is None else causal
        if cu_seqlens is not None:
            q, k, v = qkv.unbind(dim=1)
            d = q.shape[-1]
            scale = self.softmax_scale or 1.0 / math.sqrt(d)
            q_f = q.float().unsqueeze(0).permute(0, 2, 1, 3)
            k_f = k.float().unsqueeze(0).permute(0, 2, 1, 3)
            v_f = v.float().unsqueeze(0).permute(0, 2, 1, 3)
            out = torch.nn.functional.scaled_dot_product_attention(
                q_f, k_f, v_f, is_causal=causal_val, scale=scale)
            return out.permute(0, 2, 1, 3).squeeze(0).to(qkv.dtype)
        q, k, v = qkv.unbind(dim=2)
        return metal_flash_attn_func(q, k, v, causal=causal_val)

    def _flash_cross_attn_forward(self, q, kv, causal=None, **kwargs):
        causal_val = self.causal if causal is None else causal
        k, v = kv.unbind(dim=2)
        return metal_flash_attn_func(q, k, v, causal=causal_val)

    def _flash_attn_with_kvcache(q, k_cache, v_cache, k=None, v=None,
                                  rotary_cos=None, rotary_sin=None,
                                  cache_seqlens=None, softmax_scale=None,
                                  causal=True, rotary_interleaved=False,
                                  alibi_slopes=None, **kwargs):
        seqlen_offset = (cache_seqlens if isinstance(cache_seqlens, int)
                         else int(cache_seqlens.item())) if cache_seqlens is not None else 0
        if rotary_cos is not None and rotary_sin is not None:
            q = apply_rotary_torch(q, rotary_cos, rotary_sin,
                                   seqlen_offsets=seqlen_offset,
                                   interleaved=rotary_interleaved)
            if k is not None:
                k = apply_rotary_torch(k, rotary_cos, rotary_sin,
                                       seqlen_offsets=seqlen_offset,
                                       interleaved=rotary_interleaved)
        if k is not None and v is not None:
            new_len = k.shape[1]
            k_cache[:, seqlen_offset:seqlen_offset + new_len] = k
            v_cache[:, seqlen_offset:seqlen_offset + new_len] = v
            effective_len = seqlen_offset + new_len
        else:
            effective_len = seqlen_offset
        k_used = k_cache[:, :effective_len]
        v_used = v_cache[:, :effective_len]
        scale = softmax_scale or 1.0 / math.sqrt(q.shape[-1])
        q_f = q.float().permute(0, 2, 1, 3)
        k_f = k_used.float().permute(0, 2, 1, 3)
        v_f = v_used.float().permute(0, 2, 1, 3)
        out = torch.nn.functional.scaled_dot_product_attention(
            q_f, k_f, v_f, scale=scale, is_causal=False)
        return out.permute(0, 2, 1, 3).to(q.dtype)

    try:
        import vortex.model.attention as _attn
        _attn.FlashSelfAttention.forward = _flash_self_attn_forward
        _attn.FlashCrossAttention.forward = _flash_cross_attn_forward
        _attn.local_flash_attn_with_kvcache = _flash_attn_with_kvcache
        import vortex.ops as _ops
        _ops.local_flash_attn_with_kvcache = _flash_attn_with_kvcache
    except Exception as e:
        print(f"[evo2-metal] Warning: could not patch vortex attention: {e}")


# ── 7. vortex generation → default device='cpu' ───────────────────────────────

def _patch_vortex_generation():
    try:
        import vortex.model.generation as _gen
        _orig = _gen.generate

        def _cpu_generate(*, device='cpu', **kwargs):
            return _orig(device='cpu', **kwargs)

        _gen.generate = _cpu_generate
    except Exception as e:
        print(f"[evo2-metal] Warning: could not patch vortex generation: {e}")


# ── 8. evo2.scoring → default device='cpu' ────────────────────────────────────

def _patch_evo2_scoring():
    try:
        import evo2.scoring as _scoring
        _orig = _scoring.prepare_batch

        def _cpu_prepare_batch(seqs, tokenizer, prepend_bos=False, device='cpu'):
            return _orig(seqs, tokenizer, prepend_bos=prepend_bos, device='cpu')

        _scoring.prepare_batch = _cpu_prepare_batch
    except Exception as e:
        print(f"[evo2-metal] Warning: could not patch evo2.scoring: {e}")
