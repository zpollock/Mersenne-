// Broader validation of the FFT paths.
// A) squareModMersenne (fused DWT) vs reference, across a wide range of p.
// B) multiply (FFT) vs schoolbook reference at large sizes (twiddle precision).
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <string>

#include "bytearray.h"

static std::string toStr(const ByteArray& a) {
    std::string s;
    for (std::size_t i = 0; i < a.size(); ++i) s += a.getBit(i) ? '1' : '0';
    return s;
}

static ByteArray pResidue(std::size_t p, std::uint64_t seed) {
    std::string s = "1";
    std::uint64_t state = seed ? seed : 1;
    for (std::size_t i = 1; i < p; ++i) {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        s += (state & 1) ? '1' : '0';
    }
    return ByteArray::fromString(s);
}

static ByteArray randBits(std::size_t bits, std::uint64_t seed) {
    std::string s;
    std::uint64_t state = seed ? seed : 1;
    for (std::size_t i = 0; i < bits; ++i) {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        s += (state & 1) ? '1' : '0';
    }
    return ByteArray::fromString(s);
}

// Independent reference multiply from the public API.
static ByteArray refMultiply(const ByteArray& a, const ByteArray& b) {
    ByteArray acc = ByteArray::fromString("0");
    for (std::size_t j = 0; j < b.size(); ++j) {
        if (!b.getBit(b.size() - 1 - j)) continue;
        ByteArray shifted = a;
        for (std::size_t z = 0; z < j; ++z) shifted.append(0);
        acc = acc.add(shifted);
    }
    return acc;
}

static bool sameValue(const ByteArray& a, const ByteArray& b) {
    if (a.size() != b.size()) return false;
    const ByteArray diff = a.isLargerThan(b) ? a.minus(b) : b.minus(a);
    return diff.isZero();
}

int main() {
    // A) DWT fused squaring.
    const std::size_t ps[] = {61, 127, 257, 521, 1021, 1279, 2203, 3217, 5003, 8191};
    for (std::size_t p : ps) {
        for (int trial = 0; trial < 15; ++trial) {
            ByteArray s = pResidue(p, 9001 + p * 31 + trial);
            if (!sameValue(s.squareModMersenne(p), s.multiply(s).modMersenne(p))) {
                std::cout << "DWT MISMATCH p=" << p << " trial=" << trial << "\n";
                return 1;
            }
        }
        std::cout << "  DWT ok at p=" << p << "\n";
    }

    // B) FFT multiply vs independent reference at large sizes (twiddle accumulation).
    const std::size_t sizes[] = {2048, 4096, 8192, 16384, 32768, 65536};
    for (std::size_t bits : sizes) {
        ByteArray a = randBits(bits, 4242);
        ByteArray b = randBits(bits, 9919);
        if (!sameValue(a.multiply(b), refMultiply(a, b))) {
            std::cout << "MULTIPLY MISMATCH at " << bits << " bits\n";
            return 1;
        }
        std::cout << "  multiply ok at " << bits << " bits\n";
        if (bits >= 65536) break;   // reference gets slow
    }

    std::cout << "ALL PASSED\n";
    return 0;
}
