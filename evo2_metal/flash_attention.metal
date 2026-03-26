/*
 * FlashAttention — Metal Shading Language
 *
 * v1: flash_attention_forward
 *     기본 구현 (스칼라 내적)
 *
 * v2: flash_attention_forward_v2  ← 권장
 *     + float4 벡터화 내적 (d는 4의 배수여야 함; 32/64/128 전부 해당)
 *     + Multi-head 배치 디스패치 (threadgroup z축 = head 인덱스)
 *     + Causal mask (autoregressive 미래 토큰 마스킹)
 *
 * CUDA → Metal 매핑:
 *   __shared__                  → threadgroup
 *   __syncthreads()             → threadgroup_barrier(mem_flags::mem_threadgroup)
 *   blockIdx.{x,z}              → threadgroup_position_in_grid.{x,z}
 *   threadIdx.x                 → thread_position_in_threadgroup.x
 *   <<<(Tr,1,H), (Br,1,1)>>>   → dispatchThreadgroups(MTLSize(Tr,1,H), MTLSize(Br,1,1))
 *
 * 제약:
 *   d   (head_dim) ≤ 128  — 레지스터 배열 float o_acc[128]
 *   Bc  (KV tile)  ≤ 64   — 레지스터 배열 float s_scores[64]
 *   d가 4의 배수일 때 float4 최적화 자동 적용
 */

#include <metal_stdlib>
using namespace metal;

// ── v1: 기본 구현 ────────────────────────────────────────────────────────────

kernel void flash_attention_forward(
    device const float* Q     [[buffer(0)]],   // [N, d]
    device const float* K     [[buffer(1)]],
    device const float* V     [[buffer(2)]],
    device       float* O     [[buffer(3)]],
    constant uint&      N     [[buffer(4)]],
    constant uint&      d     [[buffer(5)]],
    constant uint&      Bc    [[buffer(6)]],
    constant uint&      Br    [[buffer(7)]],
    constant float&     scale [[buffer(8)]],
    threadgroup float*  tg_K  [[threadgroup(0)]],
    threadgroup float*  tg_V  [[threadgroup(1)]],
    uint3 tg_pos  [[threadgroup_position_in_grid]],
    uint3 tid_pos [[thread_position_in_threadgroup]]
) {
    const uint tg_x  = tg_pos.x;
    const uint tid   = tid_pos.x;
    const uint row_i = tg_x * Br + tid;
    const bool valid = (row_i < N);

    float o_acc[128];
    float s_scores[64];
    for (uint k = 0; k < d; k++) { o_acc[k] = 0.0f; }

    float m_i = -INFINITY;
    float l_i = 0.0f;

    const uint Tc = (N + Bc - 1) / Bc;

    for (uint j = 0; j < Tc; j++) {
        const uint kv_start = j * Bc;
        const uint kv_len   = min(Bc, N - kv_start);
        const uint total    = kv_len * d;

        for (uint idx = tid; idx < total; idx += Br) {
            const uint r = idx / d, c = idx % d;
            tg_K[r * d + c] = K[(kv_start + r) * d + c];
            tg_V[r * d + c] = V[(kv_start + r) * d + c];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (valid) {
            for (uint jj = 0; jj < kv_len; jj++) {
                float s = 0.0f;
                for (uint k = 0; k < d; k++) {
                    s += Q[row_i * d + k] * tg_K[jj * d + k];
                }
                s_scores[jj] = s * scale;
            }

            float m_new = m_i;
            for (uint jj = 0; jj < kv_len; jj++) { m_new = max(m_new, s_scores[jj]); }

            const float correction = exp(m_i - m_new);
            float l_new = l_i * correction;
            for (uint k = 0; k < d; k++) { o_acc[k] *= correction; }

            for (uint jj = 0; jj < kv_len; jj++) {
                const float p = exp(s_scores[jj] - m_new);
                l_new += p;
                for (uint k = 0; k < d; k++) { o_acc[k] += p * tg_V[jj * d + k]; }
            }
            m_i = m_new;
            l_i = l_new;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (valid && l_i > 0.0f) {
        const float inv_l = 1.0f / l_i;
        for (uint k = 0; k < d; k++) { O[row_i * d + k] = o_acc[k] * inv_l; }
    }
}


// ── v2: float4 벡터화 + Multi-head + Causal mask ─────────────────────────────

kernel void flash_attention_forward_v2(
    device const float* Q      [[buffer(0)]],  // [num_heads, N, d]  (연속 배열)
    device const float* K      [[buffer(1)]],
    device const float* V      [[buffer(2)]],
    device       float* O      [[buffer(3)]],
    constant uint&      N      [[buffer(4)]],
    constant uint&      d      [[buffer(5)]],
    constant uint&      Bc     [[buffer(6)]],
    constant uint&      Br     [[buffer(7)]],
    constant float&     scale  [[buffer(8)]],
    constant uint&      causal [[buffer(9)]],  // 1 = causal mask 적용
    threadgroup float*  tg_K   [[threadgroup(0)]],
    threadgroup float*  tg_V   [[threadgroup(1)]],
    uint3 tg_pos  [[threadgroup_position_in_grid]],   // x=Q타일, z=head 인덱스
    uint3 tid_pos [[thread_position_in_threadgroup]]
) {
    const uint head_idx   = tg_pos.z;
    const uint tile_idx   = tg_pos.x;
    const uint tid        = tid_pos.x;

    // 이 head의 데이터 시작 오프셋
    const uint base  = head_idx * N * d;
    const uint row_i = tile_idx * Br + tid;
    const bool valid = (row_i < N);

    // ── 레지스터 초기화 ──────────────────────────────────────────────────────
    float o_acc[128];
    float s_scores[64];
    for (uint k = 0; k < d; k++) { o_acc[k] = 0.0f; }

    float m_i = -INFINITY;
    float l_i = 0.0f;

    // ── Q 행을 레지스터에 미리 캐시 (device → register, 타일 루프 밖) ────────
    // 같은 Q 행을 Tc번 반복 접근하므로 레지스터에 올려 대역폭 절약
    float q_reg[128];
    if (valid) {
        const device float* q_ptr = Q + base + row_i * d;
        for (uint k = 0; k < d; k++) { q_reg[k] = q_ptr[k]; }
    }

    // ── KV 타일 루프 ─────────────────────────────────────────────────────────
    const uint Tc = (N + Bc - 1) / Bc;

    for (uint j = 0; j < Tc; j++) {
        const uint kv_start = j * Bc;
        const uint kv_len   = min(Bc, N - kv_start);

        // Causal 최적화: 이 타일의 모든 KV 행이 미래 토큰이면 통째로 건너뜀
        // (row_i < kv_start 이면 kv_start..kv_start+kv_len 전부 > row_i)
        if (causal && kv_start > row_i) {
            // 다음 타일 load가 barrier를 기다리므로 dummy barrier 필요
            threadgroup_barrier(mem_flags::mem_threadgroup);
            continue;
        }

        // ── 협력 로드: K_j, V_j → threadgroup 메모리 ────────────────────────
        // 연속 메모리 직접 복사 (행-우선 배치이므로 안전)
        const device float* k_tile_src = K + base + kv_start * d;
        const device float* v_tile_src = V + base + kv_start * d;
        const uint total = kv_len * d;
        for (uint idx = tid; idx < total; idx += Br) {
            tg_K[idx] = k_tile_src[idx];
            tg_V[idx] = v_tile_src[idx];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // ── 어텐션 스코어 S[jj] = scale * dot(q_reg, tg_K[jj]) ─────────────
        if (valid) {
            for (uint jj = 0; jj < kv_len; jj++) {
                const uint kv_row = kv_start + jj;

                // Causal: 미래 토큰을 -∞로 마스킹
                if (causal && kv_row > row_i) {
                    s_scores[jj] = -INFINITY;
                    continue;
                }

                const threadgroup float* k_ptr = tg_K + jj * d;
                float s = 0.0f;

                // float4 벡터화 내적 (d가 4의 배수이면 루프 완전 벡터화)
                uint k = 0;
                for (; k + 3 < d; k += 4) {
                    const float4 q4 = float4(q_reg[k], q_reg[k+1], q_reg[k+2], q_reg[k+3]);
                    const float4 k4 = float4(k_ptr[k], k_ptr[k+1], k_ptr[k+2], k_ptr[k+3]);
                    s += dot(q4, k4);   // Metal 내장 float4 내적
                }
                for (; k < d; k++) { s += q_reg[k] * k_ptr[k]; }  // 나머지 처리

                s_scores[jj] = s * scale;
            }

            // ── 온라인 소프트맥스 갱신 ──────────────────────────────────────
            // 1. 새로운 행 최대값
            float m_new = m_i;
            for (uint jj = 0; jj < kv_len; jj++) { m_new = max(m_new, s_scores[jj]); }

            // 2. 이전 누적값 보정 인수 (m_i=-inf → correction=0, IEEE 754 안전)
            const float correction = exp(m_i - m_new);

            // 3. P_jj = exp(s - m_new) 계산 — s_scores 레지스터를 p값으로 덮어씀
            float l_new = l_i * correction;
            for (uint jj = 0; jj < kv_len; jj++) {
                s_scores[jj] = exp(s_scores[jj] - m_new);  // p값으로 in-place 변환
                l_new += s_scores[jj];
            }

            // 4. o_acc 재스케일 + PV 누적 (float4 벡터화)
            uint k = 0;
            for (; k + 3 < d; k += 4) {
                float4 acc4 = float4(o_acc[k], o_acc[k+1], o_acc[k+2], o_acc[k+3]) * correction;
                for (uint jj = 0; jj < kv_len; jj++) {
                    const threadgroup float* v_ptr = tg_V + jj * d;
                    acc4 += s_scores[jj] * float4(v_ptr[k], v_ptr[k+1], v_ptr[k+2], v_ptr[k+3]);
                }
                o_acc[k]   = acc4.x;
                o_acc[k+1] = acc4.y;
                o_acc[k+2] = acc4.z;
                o_acc[k+3] = acc4.w;
            }
            // 나머지 차원 처리 (d가 4의 배수면 실행 안 됨)
            for (; k < d; k++) {
                o_acc[k] *= correction;
                for (uint jj = 0; jj < kv_len; jj++) {
                    o_acc[k] += s_scores[jj] * tg_V[jj * d + k];
                }
            }

            m_i = m_new;
            l_i = l_new;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // ── 정규화 후 출력 기록 ───────────────────────────────────────────────────
    if (valid && l_i > 0.0f) {
        device float* o_ptr = O + base + row_i * d;
        const float inv_l = 1.0f / l_i;
        for (uint k = 0; k < d; k++) { o_ptr[k] = o_acc[k] * inv_l; }
    }
}
