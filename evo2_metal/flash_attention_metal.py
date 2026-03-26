"""
FlashAttention on Apple Metal (via PyObjC)  —  v2

변경사항 (v2):
    - float4 벡터화: 내적/PV 누적을 Metal dot(float4, float4)로 교체
    - Multi-head 배치 디스패치: Python 루프 → GPU z축 단일 디스패치
    - Causal mask: autoregressive 미래 토큰 마스킹
    - Q 행 레지스터 캐시: device 메모리 반복 접근 제거

사용법:
    python flash_attention_metal.py          # 검증 + 벤치마크
    pip install pyobjc-framework-Metal numpy  # 의존성

CUDA → Metal 매핑:
    __shared__              → threadgroup
    __syncthreads()         → threadgroup_barrier(mem_flags::mem_threadgroup)
    blockIdx.{x,z}          → threadgroup_position_in_grid.{x,z}
    threadIdx.x             → thread_position_in_threadgroup.x
    <<<(Tr,1,H),(Br,1,1)>>> → dispatchThreadgroups(MTLSize(Tr,1,H), MTLSize(Br,1,1))
"""

import os
import struct
import numpy as np

_METAL_SOURCE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "flash_attention.metal")


# ── 레퍼런스 구현 ─────────────────────────────────────────────────────────────

def reference_attention(
    Q: np.ndarray,
    K: np.ndarray,
    V: np.ndarray,
    causal: bool = False,
) -> np.ndarray:
    """표준 Attention: softmax(QK^T / sqrt(d)) @ V — numerically stable"""
    Q = np.asarray(Q, dtype=np.float64)
    K = np.asarray(K, dtype=np.float64)
    V = np.asarray(V, dtype=np.float64)
    d = Q.shape[-1]
    scale = 1.0 / np.sqrt(d)

    S = np.matmul(Q, K.swapaxes(-2, -1)) * scale  # [..., N, N]

    if causal:
        N = Q.shape[-2]
        mask = np.triu(np.ones((N, N), dtype=bool), k=1)
        S[..., mask] = -np.inf

    S -= np.where(np.isfinite(S).any(axis=-1, keepdims=True), S.max(axis=-1, keepdims=True), 0)
    P = np.exp(S)
    P /= P.sum(axis=-1, keepdims=True)
    return (P @ V).astype(np.float32)


# ── Metal FlashAttention ──────────────────────────────────────────────────────

class MetalFlashAttention:
    """
    FlashAttention v2 — Apple Metal GPU 구현

    Parameters
    ----------
    Br : int  Q 타일 크기 (rows per threadgroup). 4의 배수, ≤ 128.
    Bc : int  KV 타일 크기. 4의 배수, ≤ 64.
    """

    def __init__(self, Br: int = 32, Bc: int = 32):
        assert 1 <= Br <= 128 and Br % 4 == 0, "Br은 4의 배수여야 하며 ≤ 128"
        assert 1 <= Bc <= 64  and Bc % 4 == 0, "Bc는 4의 배수여야 하며 ≤ 64"
        self.Br = Br
        self.Bc = Bc
        self._device   = None
        self._pipeline_v1 = None
        self._pipeline_v2 = None
        self._cmd_queue = None
        self._setup_metal()

    # ── 초기화 ──────────────────────────────────────────────────────────────

    def _setup_metal(self):
        try:
            import Metal
        except ImportError:
            print("[FlashAttention] PyObjC Metal 없음. CPU로 폴백합니다.")
            print("  → pip install pyobjc-framework-Metal")
            return

        device = Metal.MTLCreateSystemDefaultDevice()
        if device is None:
            print("[FlashAttention] Metal 디바이스를 찾을 수 없습니다.")
            return

        if not os.path.exists(_METAL_SOURCE_PATH):
            raise FileNotFoundError(f"커널 파일 없음: {_METAL_SOURCE_PATH}")

        with open(_METAL_SOURCE_PATH) as f:
            source = f.read()

        opts = Metal.MTLCompileOptions.alloc().init()
        library, err = device.newLibraryWithSource_options_error_(source, opts, None)
        if library is None:
            raise RuntimeError(f"Metal 컴파일 오류: {err}")

        def make_pipeline(name):
            fn = library.newFunctionWithName_(name)
            if fn is None:
                raise RuntimeError(f"커널 함수 '{name}'을 찾을 수 없습니다.")
            pipe, err = device.newComputePipelineStateWithFunction_error_(fn, None)
            if pipe is None:
                raise RuntimeError(f"파이프라인 생성 오류 ({name}): {err}")
            return pipe

        self._device      = device
        self._pipeline_v1 = make_pipeline("flash_attention_forward")
        self._pipeline_v2 = make_pipeline("flash_attention_forward_v2")
        self._cmd_queue   = device.newCommandQueue()

        tg_mem_kb = self.Bc * 128 * 4 * 2 / 1024
        print(f"[FlashAttention] 디바이스   : {device.name()}")
        print(f"[FlashAttention] 타일       : Br={self.Br}, Bc={self.Bc}")
        print(f"[FlashAttention] TG 메모리  : {tg_mem_kb:.0f} KB (d=128 기준, 한도 32 KB)")

    # ── 공개 인터페이스 ──────────────────────────────────────────────────────

    def forward(
        self,
        Q: np.ndarray,
        K: np.ndarray,
        V: np.ndarray,
        causal: bool = False,
    ) -> np.ndarray:
        """
        FlashAttention forward pass.

        Parameters
        ----------
        Q, K, V : np.ndarray
            float32, shape:
              [N, d]           — 단일 헤드
              [H, N, d]        — 멀티헤드 (배치 없음)
              [B, H, N, d]     — 배치 + 멀티헤드
        causal : bool
            True이면 미래 토큰 마스킹 (autoregressive 용)

        Returns
        -------
        np.ndarray  동일 shape, float32
        """
        assert Q.shape == K.shape == V.shape
        assert Q.dtype == np.float32

        if self._device is None:
            return reference_attention(Q, K, V, causal=causal)

        ndim = Q.ndim
        if ndim == 2:
            # [N, d] → [1, N, d] → dispatch → [N, d]
            return self._forward_batched(
                Q[None], K[None], V[None], causal=causal
            )[0]
        elif ndim == 3:
            # [H, N, d]
            return self._forward_batched(Q, K, V, causal=causal)
        elif ndim == 4:
            # [B, H, N, d] → [B*H, N, d] → dispatch → [B, H, N, d]
            B, H, N, d = Q.shape
            out = self._forward_batched(
                Q.reshape(B * H, N, d),
                K.reshape(B * H, N, d),
                V.reshape(B * H, N, d),
                causal=causal,
            )
            return out.reshape(B, H, N, d)
        else:
            raise ValueError(f"입력은 2~4D이어야 합니다. 현재: {ndim}D")

    @property
    def is_metal(self) -> bool:
        return self._device is not None

    # ── 배치 Metal 디스패치 ──────────────────────────────────────────────────

    def _forward_batched(
        self,
        Q: np.ndarray,   # [num_heads, N, d]
        K: np.ndarray,
        V: np.ndarray,
        causal: bool = False,
    ) -> np.ndarray:
        """num_heads개 헤드를 하나의 Metal 디스패치로 처리"""
        import Metal

        Q = np.ascontiguousarray(Q, dtype=np.float32)
        K = np.ascontiguousarray(K, dtype=np.float32)
        V = np.ascontiguousarray(V, dtype=np.float32)

        num_heads, N, d = Q.shape
        Br, Bc = self.Br, self.Bc
        scale  = np.float32(1.0 / np.sqrt(d))

        assert d <= 128, f"head_dim d={d} > 128 (커널 제한)"
        assert Bc <= 64, f"Bc={Bc} > 64 (커널 제한)"
        assert d % 4 == 0, f"d={d}는 4의 배수여야 float4 벡터화 가능"

        # ── GPU 버퍼 ────────────────────────────────────────────────────────
        opts = Metal.MTLResourceStorageModeShared
        Q_buf = self._device.newBufferWithBytes_length_options_(Q.tobytes(), Q.nbytes, opts)
        K_buf = self._device.newBufferWithBytes_length_options_(K.tobytes(), K.nbytes, opts)
        V_buf = self._device.newBufferWithBytes_length_options_(V.tobytes(), V.nbytes, opts)
        O_buf = self._device.newBufferWithLength_options_(int(Q.nbytes), opts)

        # ── 커맨드 인코딩 ────────────────────────────────────────────────────
        cmd_buf = self._cmd_queue.commandBuffer()
        enc = cmd_buf.computeCommandEncoder()
        enc.setComputePipelineState_(self._pipeline_v2)

        enc.setBuffer_offset_atIndex_(Q_buf, 0, 0)
        enc.setBuffer_offset_atIndex_(K_buf, 0, 1)
        enc.setBuffer_offset_atIndex_(V_buf, 0, 2)
        enc.setBuffer_offset_atIndex_(O_buf, 0, 3)

        # 상수 (setBytes = 인라인 전달, 별도 버퍼 할당 불필요)
        enc.setBytes_length_atIndex_(struct.pack("I", N),            4, 4)
        enc.setBytes_length_atIndex_(struct.pack("I", d),            4, 5)
        enc.setBytes_length_atIndex_(struct.pack("I", Bc),           4, 6)
        enc.setBytes_length_atIndex_(struct.pack("I", Br),           4, 7)
        enc.setBytes_length_atIndex_(struct.pack("f", scale),        4, 8)
        enc.setBytes_length_atIndex_(struct.pack("I", int(causal)),  4, 9)

        # threadgroup 메모리: tg_K (index 0), tg_V (index 1)
        tg_tile_bytes = int(Bc * d * 4)
        enc.setThreadgroupMemoryLength_atIndex_(tg_tile_bytes, 0)
        enc.setThreadgroupMemoryLength_atIndex_(tg_tile_bytes, 1)

        # 디스패치: x=Q타일, z=head 인덱스 → 모든 head를 한 번에 처리
        Tr = (N + Br - 1) // Br
        enc.dispatchThreadgroups_threadsPerThreadgroup_(
            Metal.MTLSizeMake(Tr, 1, num_heads),
            Metal.MTLSizeMake(Br, 1, 1),
        )
        enc.endEncoding()

        cmd_buf.commit()
        cmd_buf.waitUntilCompleted()

        err = cmd_buf.error()
        if err is not None:
            raise RuntimeError(f"Metal 커맨드 오류: {err}")

        # ── 결과 읽기 ────────────────────────────────────────────────────────
        raw = O_buf.contents().as_buffer(int(Q.nbytes))
        return np.frombuffer(raw, dtype=np.float32).copy().reshape(num_heads, N, d)


# ── 검증 ─────────────────────────────────────────────────────────────────────

def run_validation():
    fa = MetalFlashAttention(Br=32, Bc=32)
    backend = "Metal GPU (v2)" if fa.is_metal else "CPU 폴백"

    print(f"\n{'='*62}")
    print(f"  FlashAttention 검증  —  {backend}")
    print(f"{'='*62}")

    # ── Full attention 검증 ──────────────────────────────────────────────────
    print(f"\n  [Full Attention]")
    print(f"  {'N':>5} {'d':>4} {'heads':>5}  {'최대오차':>10}  {'상태':>6}  설명")
    print(f"  {'-'*5} {'-'*4} {'-'*5}  {'-'*10}  {'-'*6}  {'-'*20}")

    full_cases = [
        (32,   64, 1, "기본"),
        (64,   64, 1, "N=64"),
        (128,  64, 1, "N=128"),
        (256,  64, 1, "N=256"),
        (1024, 64, 1, "N=1024"),
        (128, 128, 1, "d=128"),
        (100,  64, 1, "N이 Br 배수 아닌 경우"),
        (50,   64, 1, "N < Bc"),
        (128,  64, 8, "8-head"),
        (256,  64, 4, "4-head B=2", ),
    ]

    all_pass = True
    rng = np.random.default_rng(42)

    for row in full_cases:
        if len(row) == 4:
            N, d, H, desc = row
            B = 1
        else:
            N, d, H, B, desc = row
        shape = (B, H, N, d) if B > 1 else (H, N, d)
        Q = rng.standard_normal(shape).astype(np.float32)
        K = rng.standard_normal(shape).astype(np.float32)
        V = rng.standard_normal(shape).astype(np.float32)

        O_ref = reference_attention(Q, K, V)
        O_fa  = fa.forward(Q, K, V)

        max_err = float(np.abs(O_fa - O_ref).max())
        ok = max_err < 1e-4
        all_pass = all_pass and ok
        status = "PASS" if ok else "FAIL"
        print(f"  {N:>5} {d:>4} {H:>5}  {max_err:>10.2e}  {status:>6}  {desc}")

    # ── Causal attention 검증 ────────────────────────────────────────────────
    print(f"\n  [Causal Attention]")
    print(f"  {'N':>5} {'d':>4} {'heads':>5}  {'최대오차':>10}  {'상태':>6}  설명")
    print(f"  {'-'*5} {'-'*4} {'-'*5}  {'-'*10}  {'-'*6}  {'-'*20}")

    causal_cases = [
        (64,   64, 1, "causal 기본"),
        (128,  64, 1, "causal N=128"),
        (256,  64, 1, "causal N=256"),
        (1024, 64, 1, "causal N=1024"),
        (100,  64, 1, "causal N 비정렬"),
        (128,  64, 8, "causal 8-head"),
    ]

    for N, d, H, desc in causal_cases:
        Q = rng.standard_normal((H, N, d)).astype(np.float32)
        K = rng.standard_normal((H, N, d)).astype(np.float32)
        V = rng.standard_normal((H, N, d)).astype(np.float32)

        O_ref = reference_attention(Q, K, V, causal=True)
        O_fa  = fa.forward(Q, K, V, causal=True)

        max_err = float(np.abs(O_fa - O_ref).max())
        ok = max_err < 1e-4
        all_pass = all_pass and ok
        status = "PASS" if ok else "FAIL"
        print(f"  {N:>5} {d:>4} {H:>5}  {max_err:>10.2e}  {status:>6}  {desc}")

    print(f"\n{'='*62}")
    print(f"  결과: {'전체 PASS ✓' if all_pass else '일부 FAIL ✗'}")
    print(f"{'='*62}\n")
    return all_pass


def run_perf_benchmark():
    """v1(스칼라) vs v2(float4+배치) 비교 + multi-head 처리량"""
    import time
    import Metal

    fa = MetalFlashAttention(Br=32, Bc=32)
    if not fa.is_metal:
        return

    print(f"{'='*65}")
    print("  벤치마크  —  v2 (float4 + 배치 디스패치)  vs  v1 (스칼라)")
    print(f"{'='*65}")

    # v1 파이프라인을 직접 사용하는 헬퍼
    def run_v1(Q, K, V):
        """v1 단일 헤드 실행 (비교용)"""
        N, d = Q.shape
        Br, Bc = fa.Br, fa.Bc
        scale = np.float32(1.0 / np.sqrt(d))
        opts = Metal.MTLResourceStorageModeShared
        Q_b = fa._device.newBufferWithBytes_length_options_(Q.tobytes(), Q.nbytes, opts)
        K_b = fa._device.newBufferWithBytes_length_options_(K.tobytes(), K.nbytes, opts)
        V_b = fa._device.newBufferWithBytes_length_options_(V.tobytes(), V.nbytes, opts)
        O_b = fa._device.newBufferWithLength_options_(Q.nbytes, opts)
        cmd = fa._cmd_queue.commandBuffer()
        enc = cmd.computeCommandEncoder()
        enc.setComputePipelineState_(fa._pipeline_v1)
        for i, buf in enumerate([Q_b, K_b, V_b, O_b]):
            enc.setBuffer_offset_atIndex_(buf, 0, i)
        enc.setBytes_length_atIndex_(struct.pack("I", N),     4, 4)
        enc.setBytes_length_atIndex_(struct.pack("I", d),     4, 5)
        enc.setBytes_length_atIndex_(struct.pack("I", Bc),    4, 6)
        enc.setBytes_length_atIndex_(struct.pack("I", Br),    4, 7)
        enc.setBytes_length_atIndex_(struct.pack("f", scale), 4, 8)
        enc.setThreadgroupMemoryLength_atIndex_(Bc * d * 4, 0)
        enc.setThreadgroupMemoryLength_atIndex_(Bc * d * 4, 1)
        Tr = (N + Br - 1) // Br
        enc.dispatchThreadgroups_threadsPerThreadgroup_(
            Metal.MTLSizeMake(Tr, 1, 1), Metal.MTLSizeMake(Br, 1, 1))
        enc.endEncoding()
        cmd.commit()
        cmd.waitUntilCompleted()

    def bench(fn, warmup=5, iters=20):
        for _ in range(warmup): fn()
        t0 = time.perf_counter()
        for _ in range(iters): fn()
        return (time.perf_counter() - t0) / iters * 1e3

    rng = np.random.default_rng(0)

    # 단일 헤드: v1 vs v2
    print(f"\n  단일 헤드 (H=1)  —  v1 스칼라 vs v2 float4")
    print(f"  {'N':>6} {'d':>4}  {'v1 ms':>8}  {'v2 ms':>8}  {'속도향상':>8}")
    print(f"  {'-'*6} {'-'*4}  {'-'*8}  {'-'*8}  {'-'*8}")
    for N, d in [(512, 64), (1024, 64), (2048, 64), (1024, 128)]:
        Q = rng.standard_normal((N, d)).astype(np.float32)
        K = rng.standard_normal((N, d)).astype(np.float32)
        V = rng.standard_normal((N, d)).astype(np.float32)
        t1 = bench(lambda: run_v1(Q, K, V))
        t2 = bench(lambda: fa.forward(Q, K, V))
        print(f"  {N:>6} {d:>4}  {t1:>8.2f}  {t2:>8.2f}  {t1/t2:>7.2f}x")

    # 멀티헤드 배치 디스패치
    print(f"\n  멀티헤드 배치 디스패치")
    print(f"  {'N':>6} {'d':>4} {'H':>3}  {'ms/call':>8}  {'TFLOP/s':>9}  {'설명'}")
    print(f"  {'-'*6} {'-'*4} {'-'*3}  {'-'*8}  {'-'*9}  {'-'*20}")
    for N, d, H, desc in [
        (512,  64,  1, "단일 헤드"),
        (512,  64,  8, "8-head (GPT-2 base)"),
        (512,  64, 12, "12-head (GPT-2 large)"),
        (512,  64, 16, "16-head"),
        (1024, 64,  8, "N=1024, 8-head"),
        (2048, 64,  8, "N=2048, 8-head"),
    ]:
        Q = rng.standard_normal((H, N, d)).astype(np.float32)
        K = rng.standard_normal((H, N, d)).astype(np.float32)
        V = rng.standard_normal((H, N, d)).astype(np.float32)
        t = bench(lambda: fa.forward(Q, K, V))
        flops = H * 4 * N * N * d
        tflops = flops / (t / 1e3) / 1e12
        print(f"  {N:>6} {d:>4} {H:>3}  {t:>8.2f}  {tflops:>9.4f}  {desc}")

    print(f"\n{'='*65}\n")


if __name__ == "__main__":
    ok = run_validation()
    if ok:
        run_perf_benchmark()
