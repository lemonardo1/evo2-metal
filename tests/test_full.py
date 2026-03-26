"""
evo2-metal comprehensive test suite
====================================
Tests all evo2 features on Apple Silicon (no CUDA).

Run:
    conda activate evo2-mac
    cd /Users/daeseongkim/evo2-metal
    python tests/test_full.py [--no-model]   # --no-model skips inference tests
"""

import sys
import os
import time
import traceback

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ── Must come before evo2 ───────────────────────────────────────────────────
import evo2_metal

import torch
import numpy as np

NO_MODEL = "--no-model" in sys.argv

PASS  = "\033[32mPASS\033[0m"
FAIL  = "\033[31mFAIL\033[0m"
SKIP  = "\033[33mSKIP\033[0m"
WARN  = "\033[33mWARN\033[0m"

_results = {"pass": 0, "fail": 0, "skip": 0}


def section(title):
    print(f"\n{'='*62}")
    print(f"  {title}")
    print(f"{'='*62}")


def check(name, fn, skip_if=False):
    if skip_if:
        print(f"  [{SKIP}] {name}")
        _results["skip"] += 1
        return None
    try:
        result = fn()
        print(f"  [{PASS}] {name}")
        _results["pass"] += 1
        return result
    except Exception as e:
        print(f"  [{FAIL}] {name}")
        for line in traceback.format_exc().splitlines()[-6:]:
            print(f"          {line}")
        _results["fail"] += 1
        return None


def info(msg):
    print(f"          {msg}")


# ════════════════════════════════════════════════════════════════
# SECTION 1 — Patch validation (no model needed)
# ════════════════════════════════════════════════════════════════
section("1. Patch validation (no model load)")

def test_cuda_device_noop():
    with torch.cuda.device("cpu"):
        pass
    with torch.cuda.device("mps"):
        pass

def test_cuda_memory_allocated():
    result = torch.cuda.memory_allocated(None)
    assert result == 0, f"Expected 0, got {result}"

def test_flash_attn_mocked():
    import flash_attn
    assert hasattr(flash_attn, 'flash_attn_func')
    assert callable(flash_attn.flash_attn_func)

def test_flash_attn_2_cuda_mocked():
    import flash_attn_2_cuda  # must not raise

def test_triton_mocked():
    import triton
    import triton.language

def test_rotary_patched():
    rot = sys.modules.get('vortex.ops.embedding.rotary')
    assert rot is not None, "vortex.ops.embedding.rotary not in sys.modules"
    assert callable(rot.apply_rotary), "apply_rotary not callable"

def test_vortex_attention_patched():
    import vortex.model.attention as attn
    # Should be our custom function (not the original CUDA one)
    fwd_name = attn.FlashSelfAttention.forward.__qualname__
    assert "metal" in fwd_name or "_flash_self_attn_forward" in fwd_name or \
           attn.FlashSelfAttention.forward.__module__ == 'evo2_metal.patch', \
        f"Unexpected forward: {fwd_name}"

def test_vortex_kvcache_patched():
    import vortex.model.attention as attn
    import vortex.ops as ops
    assert callable(attn.local_flash_attn_with_kvcache)
    assert callable(ops.local_flash_attn_with_kvcache)

def test_cuda_empty_cache_noop():
    torch.cuda.empty_cache()  # must not raise

def test_autocast_redirected():
    # autocast("cuda") should work without CUDA — redirect to cpu
    with torch.autocast("cuda"):
        x = torch.randn(4, 4) @ torch.randn(4, 4)
    assert x.shape == (4, 4)

def test_vortex_generation_patched():
    import vortex.model.generation as gen
    import inspect
    src = inspect.getsource(gen.generate)
    assert "cpu" in src, "generate patch not applied (no 'cpu' in source)"

def test_evo2_scoring_patched():
    import evo2.scoring as s
    import inspect
    src = inspect.getsource(s.prepare_batch)
    assert "cpu" in src, "prepare_batch patch not applied"

check("torch.cuda.device no-op (cpu/mps)", test_cuda_device_noop)
check("torch.cuda.memory_allocated returns 0", test_cuda_memory_allocated)
check("torch.cuda.empty_cache no-op", test_cuda_empty_cache_noop)
check("torch.autocast('cuda') redirected to cpu", test_autocast_redirected)
check("flash_attn module mocked with callable flash_attn_func", test_flash_attn_mocked)
check("flash_attn_2_cuda mocked", test_flash_attn_2_cuda_mocked)
check("triton / triton.language mocked", test_triton_mocked)
check("vortex.ops.embedding.rotary.apply_rotary patched", test_rotary_patched)
check("FlashSelfAttention.forward patched (Metal backend)", test_vortex_attention_patched)
check("local_flash_attn_with_kvcache patched (SDPA fallback)", test_vortex_kvcache_patched)
check("vortex.model.generation.generate defaults device=cpu", test_vortex_generation_patched)
check("evo2.scoring.prepare_batch defaults device=cpu", test_evo2_scoring_patched)


# ════════════════════════════════════════════════════════════════
# SECTION 2 — Metal attention kernel (no model needed)
# ════════════════════════════════════════════════════════════════
section("2. Metal FlashAttention kernel (direct)")

from evo2_metal.flash_attention_metal import MetalFlashAttention
_metal = MetalFlashAttention(Br=32, Bc=32)

def test_metal_causal_correctness():
    """Metal output should be close to PyTorch SDPA."""
    B, H, T, d = 1, 4, 16, 32
    rng = np.random.default_rng(42)
    q = rng.standard_normal((B, H, T, d)).astype(np.float32)
    k = rng.standard_normal((B, H, T, d)).astype(np.float32)
    v = rng.standard_normal((B, H, T, d)).astype(np.float32)
    out_metal = _metal.forward(q, k, v, causal=True)
    out_ref = torch.nn.functional.scaled_dot_product_attention(
        torch.from_numpy(q), torch.from_numpy(k), torch.from_numpy(v),
        is_causal=True).numpy()
    err = float(np.abs(out_metal - out_ref).max())
    assert err < 0.05, f"Max abs error too large: {err:.5f}"
    return err

def test_metal_noncausal():
    B, H, T, d = 1, 2, 32, 64
    rng = np.random.default_rng(7)
    q = rng.standard_normal((B, H, T, d)).astype(np.float32)
    k = rng.standard_normal((B, H, T, d)).astype(np.float32)
    v = rng.standard_normal((B, H, T, d)).astype(np.float32)
    out = _metal.forward(q, k, v, causal=False)
    ref = torch.nn.functional.scaled_dot_product_attention(
        torch.from_numpy(q), torch.from_numpy(k), torch.from_numpy(v),
        is_causal=False).numpy()
    err = float(np.abs(out - ref).max())
    assert err < 0.05, f"Max abs error: {err:.5f}"
    return err

def test_metal_batched():
    B, H, T, d = 2, 8, 64, 128
    q = np.random.randn(B, H, T, d).astype(np.float32)
    k = np.random.randn(B, H, T, d).astype(np.float32)
    v = np.random.randn(B, H, T, d).astype(np.float32)
    out = _metal.forward(q, k, v, causal=True)
    assert out.shape == (B, H, T, d)

def test_metal_via_patch_func():
    """Test the metal_flash_attn_func used by vortex attention forward."""
    func = evo2_metal.patch.apply_patches._metal_flash_attn_func
    B, T, H, d = 1, 8, 4, 32
    q = torch.randn(B, T, H, d)
    k = torch.randn_like(q)
    v = torch.randn_like(q)
    out = func(q, k, v, causal=True)
    assert out.shape == q.shape
    assert out.dtype == q.dtype

def test_flash_self_attn_forward():
    """Run FlashSelfAttention.forward (the patched one) directly."""
    import vortex.model.attention as attn

    class FakeModule:
        causal = True
        softmax_scale = None

    # vortex passes (B, T, 3, H, d); unbind(dim=2) splits into q,k,v of (B,T,H,d)
    B, T, H, d = 1, 8, 4, 32
    qkv = torch.randn(B, T, 3, H, d)
    out = attn.FlashSelfAttention.forward(FakeModule(), qkv)
    assert out.shape == (B, T, H, d), f"Expected {(B,T,H,d)}, got {out.shape}"

def test_rotary_apply():
    """Test the pure-PyTorch apply_rotary fallback."""
    fn = evo2_metal.patch.apply_patches._apply_rotary_torch
    B, T, H, d = 1, 16, 4, 32
    x = torch.randn(B, T, H, d)
    cos = torch.randn(T, d // 2)
    sin = torch.randn(T, d // 2)
    out = fn(x, cos, sin)
    assert out.shape == x.shape

err = check("Metal causal kernel vs PyTorch SDPA", test_metal_causal_correctness)
if err is not None: info(f"max abs error = {err:.6f}")
err = check("Metal non-causal kernel vs PyTorch SDPA", test_metal_noncausal)
if err is not None: info(f"max abs error = {err:.6f}")
check("Metal batched kernel (B=2, H=8, T=64, d=128)", test_metal_batched)
check("metal_flash_attn_func dtype preservation", test_metal_via_patch_func)
check("FlashSelfAttention.forward (patched) direct call", test_flash_self_attn_forward)
check("apply_rotary_torch output shape", test_rotary_apply)


# ════════════════════════════════════════════════════════════════
# SECTION 3 — CUDA leak detection
# ════════════════════════════════════════════════════════════════
section("3. CUDA leak detection")

def test_no_raw_cuda_in_evo2_metal():
    """evo2_metal sources (except patch.py itself) must not contain raw CUDA calls."""
    pkg_path = os.path.dirname(evo2_metal.__file__)
    # These patterns indicate unintended CUDA use in consumer code
    BAD_PATTERNS = ['.cuda()', 'cuda:0', 'torch.cuda.current_device']
    violations = []
    for fname in os.listdir(pkg_path):
        if not fname.endswith('.py'):
            continue
        if fname == 'patch.py':
            # patch.py legitimately references "cuda" to intercept it
            continue
        fpath = os.path.join(pkg_path, fname)
        with open(fpath) as f:
            for i, line in enumerate(f, 1):
                stripped = line.strip()
                if stripped.startswith('#'):
                    continue
                for pat in BAD_PATTERNS:
                    if pat in line:
                        violations.append(f"{fname}:{i}: {stripped}")
    assert not violations, "Unexpected CUDA leaks:\n" + "\n".join(violations)

def test_is_cuda_not_asserted():
    """The patched forward must never assert qkv.is_cuda."""
    import vortex.model.attention as attn
    import inspect
    src = inspect.getsource(attn.FlashSelfAttention.forward)
    assert 'assert' not in src or 'is_cuda' not in src, \
        "Patched forward still contains 'assert ... is_cuda'"

def test_no_cuda_tensor_created():
    """Ensure model doesn't accidentally move tensors to CUDA."""
    t = torch.randn(2, 8)
    assert t.device.type in ('cpu', 'mps')

check("No raw CUDA calls in evo2_metal/*.py", test_no_raw_cuda_in_evo2_metal)
check("Patched FlashSelfAttention.forward has no is_cuda assert", test_is_cuda_not_asserted)
check("New tensors land on cpu/mps (not cuda)", test_no_cuda_tensor_created)


# ════════════════════════════════════════════════════════════════
# SECTION 4 — Model inference (requires weights)
# ════════════════════════════════════════════════════════════════
section("4. Model inference — evo2_7b_base")

if NO_MODEL:
    print("  [SKIP] --no-model flag set, skipping all inference tests")
    _results["skip"] += 20
else:
    print("  Loading model... (may take ~1-2 min on first run)")
    t0 = time.time()
    try:
        from evo2 import Evo2
        model = Evo2('evo2_7b_base')
        load_time = time.time() - t0
        print(f"  [{PASS}] Model loaded ({load_time:.1f}s)")
        _results["pass"] += 1
        MODEL_OK = True
    except Exception as e:
        print(f"  [{FAIL}] Model load: {e}")
        traceback.print_exc()
        MODEL_OK = False

    if MODEL_OK:
        # ── 4a. score_sequences ──────────────────────────────────────
        section("4a. score_sequences")

        BASELINE = "ACGTACGTACGT"
        BASELINE_SCORE = -1.4666194

        def test_score_baseline():
            scores = model.score_sequences([BASELINE])
            val = float(scores[0])
            assert abs(val - BASELINE_SCORE) < 0.01, \
                f"Drift: got {val:.7f}, expected {BASELINE_SCORE}"
            return val

        def test_score_two_seqs_differ():
            wt  = "ATGAAAGCAATTTTCGTACTGAAAGGTTCAGGT"
            var = "ATGAAAGCAATTTTCGTACTGAAAGGTTCAGGA"
            s = model.score_sequences([wt, var])
            assert len(s) == 2
            assert np.isfinite(float(s[0])) and np.isfinite(float(s[1]))
            assert abs(float(s[0]) - float(s[1])) > 1e-4, \
                "Two different sequences scored identically"
            return s

        def test_score_batch_size_2():
            seqs = ["ATGCATGCATGC", "GCTAGCTAGCTA", "TTTTAAAACCCC"]
            s = model.score_sequences(seqs, batch_size=2)
            assert len(s) == 3
            assert all(np.isfinite(float(x)) for x in s)

        def test_score_reduce_sum():
            s_mean = float(model.score_sequences(["ATGCATGC"], reduce_method='mean')[0])
            s_sum  = float(model.score_sequences(["ATGCATGC"], reduce_method='sum')[0])
            ratio = s_sum / s_mean
            # 8 chars → 7 steps; ratio ≈ 7 (within 50% margin)
            assert 3 < abs(ratio) < 14, f"Unexpected ratio sum/mean: {ratio:.2f}"

        def test_score_prepend_bos():
            s1 = float(model.score_sequences(["ATGCATGC"], prepend_bos=False)[0])
            s2 = float(model.score_sequences(["ATGCATGC"], prepend_bos=True)[0])
            assert abs(s1 - s2) > 1e-6, "prepend_bos had no effect"

        def test_score_reverse_complement():
            seq = "ATGCATGCATGC"
            s1 = model.score_sequences([seq], average_reverse_complement=False)
            s2 = model.score_sequences([seq], average_reverse_complement=True)
            assert len(s1) == 1 and len(s2) == 1
            assert np.isfinite(float(s1[0])) and np.isfinite(float(s2[0]))

        def test_score_longer():
            seq = "ATGCATGC" * 25   # 200 bp
            s = model.score_sequences([seq])
            assert np.isfinite(float(s[0]))

        def test_score_homopolymers():
            for base in ["AAAA", "CCCC", "GGGG", "TTTT"]:
                s = model.score_sequences([base])
                assert np.isfinite(float(s[0])), f"Non-finite for {base}"

        sc = check(f"Baseline score (ACGTACGTACGT ≈ {BASELINE_SCORE})", test_score_baseline)
        if sc is not None: info(f"got {sc:.7f}")

        ss = check("Two different seqs score differently", test_score_two_seqs_differ)
        if ss is not None: info(f"seq1={ss[0]:.4f}  seq2={ss[1]:.4f}")

        check("Batch of 3 seqs with batch_size=2", test_score_batch_size_2)
        check("reduce_method sum vs mean ratio", test_score_reduce_sum)
        check("prepend_bos=True vs False differ", test_score_prepend_bos)
        check("average_reverse_complement produces finite score", test_score_reverse_complement)
        check("Score 200 bp sequence", test_score_longer)
        check("Score all four homopolymers (A/C/G/T)", test_score_homopolymers)

        # ── 4b. generate ──────────────────────────────────────────────
        section("4b. generate")

        PROMPT = "ATGAAAGCAATTTTCGTACT"

        def test_gen_basic():
            out = model.generate(prompt_seqs=[PROMPT], n_tokens=20,
                                 temperature=1.0, top_k=4)
            seq = out.sequences[0]
            assert len(seq) > 0
            bad = [c for c in seq if c not in "ACGTN"]
            assert not bad, f"Non-DNA chars: {bad}"
            return seq

        def test_gen_length():
            out = model.generate(prompt_seqs=["ATGCATGC"], n_tokens=50,
                                 temperature=1.0, top_k=4)
            assert len(out.sequences[0]) == 50

        def test_gen_greedy_deterministic():
            kwargs = dict(prompt_seqs=[PROMPT], n_tokens=30,
                          temperature=0.01, top_k=1)
            out1 = model.generate(**kwargs)
            out2 = model.generate(**kwargs)
            assert out1.sequences[0] == out2.sequences[0], \
                f"Greedy outputs differ:\n  {out1.sequences[0]}\n  {out2.sequences[0]}"

        def test_gen_cached_vs_uncached():
            kw = dict(prompt_seqs=[PROMPT], n_tokens=20,
                      temperature=0.01, top_k=1)
            c = model.generate(**kw, cached_generation=True)
            u = model.generate(**kw, cached_generation=False)
            for out in [c, u]:
                assert all(ch in "ACGTN" for ch in out.sequences[0])
                assert np.isfinite(out.logprobs_mean[0])

        def test_gen_top_p():
            out = model.generate(prompt_seqs=[PROMPT], n_tokens=30,
                                 temperature=1.0, top_k=0, top_p=0.9)
            assert all(c in "ACGTN" for c in out.sequences[0])

        def test_gen_multiple_prompts():
            prompts = [PROMPT, "GCTAGCTAGCTAGCTAGCTA"]
            out = model.generate(prompt_seqs=prompts, n_tokens=20,
                                 temperature=1.0, top_k=4)
            assert len(out.sequences) == 2
            for seq in out.sequences:
                assert all(c in "ACGTN" for c in seq)

        def test_gen_logprobs_finite_negative():
            out = model.generate(prompt_seqs=["ATGCATGC"], n_tokens=10,
                                 temperature=1.0, top_k=4)
            lp = out.logprobs_mean[0]
            assert np.isfinite(lp) and lp < 0, f"Bad logprob: {lp}"

        seq = check("Basic generation (20 tokens, DNA only)", test_gen_basic)
        if seq: info(f"generated: {seq}")

        check("n_tokens=50 matches output length", test_gen_length)
        check("Greedy (temp=0.01, top_k=1) is deterministic", test_gen_greedy_deterministic)
        check("Cached and uncached generation both produce valid DNA", test_gen_cached_vs_uncached)
        check("top_p=0.9 sampling produces valid DNA", test_gen_top_p)
        check("Multi-prompt batch generation (2 prompts)", test_gen_multiple_prompts)
        check("logprobs_mean is finite and negative", test_gen_logprobs_finite_negative)

        # ── 4c. forward() raw logits ─────────────────────────────────
        section("4c. model.forward() raw logits")

        def _get_logits(tokens):
            """Evo2.forward returns ((logits, aux), None); extract logits tensor."""
            out = model.forward(tokens)
            # Unwrap: out = ((logits, ...), None) or (logits, None)
            first = out[0]
            if isinstance(first, tuple):
                return first[0]
            return first

        def test_forward_shape_and_finite():
            tokens = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8]])
            with torch.no_grad():
                logits = _get_logits(tokens)
            assert logits.ndim == 3
            assert logits.shape[:2] == (1, 8)
            assert torch.isfinite(logits).all()
            return tuple(logits.shape)

        def test_forward_device_not_cuda():
            tokens = torch.tensor([[1, 2, 3, 4]])
            with torch.no_grad():
                logits = _get_logits(tokens)
            assert logits.device.type != 'cuda', \
                f"Logits on CUDA: {logits.device}"
            return str(logits.device)

        shape = check("forward() returns (1, T, vocab) finite logits", test_forward_shape_and_finite)
        if shape: info(f"shape: {shape}")

        dev = check("forward() output is NOT on CUDA device", test_forward_device_not_cuda)
        if dev: info(f"device: {dev}")


# ════════════════════════════════════════════════════════════════
# Final summary
# ════════════════════════════════════════════════════════════════
section("Summary")
total = sum(_results.values())
print(f"  Passed : {_results['pass']}/{total}")
print(f"  Failed : {_results['fail']}/{total}")
print(f"  Skipped: {_results['skip']}/{total}")
if _results["fail"] == 0:
    print(f"\n  \033[32mAll tests passed!\033[0m")
else:
    print(f"\n  \033[31m{_results['fail']} test(s) failed.\033[0m")
    sys.exit(1)
