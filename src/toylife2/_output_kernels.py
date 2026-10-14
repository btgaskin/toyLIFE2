"""Exact uint32 Philox and count-based apparatus for resident decision kernels."""

# pyright: reportCallIssue=false, reportAttributeAccessIssue=false, reportInvalidTypeForm=false, reportOperatorIssue=false
from __future__ import annotations

import quadrants as qd


@qd.func
def multiply_high(a, b):
    # 16-bit limbs keep the implementation portable to Metal without uint64.
    mask = qd.u32(65535)
    a0, a1 = a & mask, a >> 16
    b0, b1 = b & mask, b >> 16
    low = a0 * b0
    middle = a1 * b0 + (low >> 16)
    other = a0 * b1 + (middle & mask)
    return a1 * b1 + (middle >> 16) + (other >> 16)


@qd.func
def philox_word(index_lo, index_hi, key0, key1, attempt):
    c0, c1, c2, c3 = index_lo, index_hi, attempt, qd.u32(0)
    k0, k1 = key0, key1
    for _ in range(10):
        m0, m1 = qd.u32(0xD2511F53), qd.u32(0xCD9E8D57)
        n0 = multiply_high(m1, c2) ^ c1 ^ k0
        n1 = m1 * c2
        n2 = multiply_high(m0, c0) ^ c3 ^ k1
        n3 = m0 * c0
        c0, c1, c2, c3 = n0, n1, n2, n3
        k0 += qd.u32(0x9E3779B9)
        k1 += qd.u32(0xBB67AE85)
    return c0


@qd.func
def binary_action(first, second, key0, key1, index_lo, index_hi, rule, swap):
    a, b = first, second
    if rule == 2:  # majority
        a, b = qd.cast(first > second, qd.i32), qd.cast(second > first, qd.i32)
    if a + b == 0:
        a, b = 1, 1
    upper = qd.cast(a + b, qd.u32)
    threshold = (qd.u32(0) - upper) % upper
    attempt = qd.u32(0)
    word = philox_word(index_lo, index_hi, key0, key1, attempt)
    while word < threshold:
        attempt += qd.u32(1)
        word = philox_word(index_lo, index_hi, key0, key1, attempt)
    return qd.cast(word % upper < qd.cast(b, qd.u32), qd.i32) ^ swap


@qd.kernel(fastcache=True)
def sample_actions(
    counts: qd.types.NDArray[qd.i32, 2],
    keys: qd.types.NDArray[qd.u32, 2],
    ids: qd.types.NDArray[qd.u32, 2],
    output: qd.types.NDArray[qd.i32, 1],
    rule: qd.i32,
    swap: qd.i32,
):
    for lane in range(output.shape[0]):
        output[lane] = binary_action(
            counts[lane, 0],
            counts[lane, 1],
            keys[lane, 0],
            keys[lane, 1],
            ids[lane, 0],
            ids[lane, 1],
            rule,
            swap,
        )
