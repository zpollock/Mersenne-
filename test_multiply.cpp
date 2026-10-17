// Standalone test: verify the FFT-based multiply (engaged above 1024 bits)
// produces the exact same result as an independent schoolbook reference
// built only from the public API.
#include <cstdint>
#include <cstdlib>
#include <iostream>

#include "bytearray.h"

// Deterministic pseudo-random bit array of `bits` length.
static ByteArray randomBits(std::size_t bits, std::uint64_t seed) {
    ByteArray r;
    std::uint64_t state = seed ? seed : 1;
    for (std::size_t i = 0; i < bits; ++i) {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        r.append(static_cast<std::uint8_t>(state & 1));
    }
    return r;
}

// Independent reference multiply using only the public API: shift-and-add on
// the MSB-first bits read via getBit().
static ByteArray refMultiply(const ByteArray& a, const ByteArray& b) {
    ByteArray acc;
    acc = ByteArray::fromString("0");
    for (std::size_t j = 0; j < b.size(); ++j) {
        // 2^j place of b is the bit at MSB position (size-1-j).
        if (!b.getBit(b.size() - 1 - j)) continue;
        ByteArray shifted = a;
        for (std::size_t z = 0; z < j; ++z) shifted.append(0);
        acc = acc.add(shifted);
    }
    return acc;
}

// True if a and b hold the same value.
static bool sameValue(const ByteArray& a, const ByteArray& b) {
    if (a.size() != b.size()) return false;
    const ByteArray diff = a.isLargerThan(b) ? a.minus(b) : b.minus(a);
    return diff.isZero();
}

int main() {
    // Cross-check the FFT path (>=1024-bit operands) against the reference.
    const std::size_t sizes[] = {1024, 1100, 1500, 2048, 3000, 5000};
    for (std::size_t bits : sizes) {
        for (int trial = 0; trial < 3; ++trial) {
            ByteArray a = randomBits(bits, 101 + trial);
            ByteArray b = randomBits(bits, 777 + trial * 13);

            const ByteArray prod = a.multiply(b);   // FFT/dispatch
            const ByteArray expect = refMultiply(a, b);

            if (!sameValue(prod, expect)) {
                std::cout << "VALUE MISMATCH at " << bits
                          << " bits, trial " << trial << "\n";
                return 1;
            }
        }
    }

    // Square a large value (the operation Lucas-Lehmer uses).
    const std::size_t big = 4096;
    ByteArray s = randomBits(big, 55);
    if (!sameValue(s.multiply(s), refMultiply(s, s))) {
        std::cout << "SQUARE MISMATCH at " << big << " bits\n";
        return 1;
    }

    std::cout << "All FFT multiply cross-checks passed.\n";
    return 0;
}
