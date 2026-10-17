// Calibration: find the correct DWT weight/deweight convention.
// Use p=13, s small so s^2 < 2^13 (no folding). squareModMersenne(13) must
// return exactly 25 for s=5.
#include <cmath>
#include <cstdint>
#include <iostream>
#include <string>
#include <vector>

#include "bytearray.h"

static std::string toStr(const ByteArray& a) {
    std::string s;
    for (std::size_t i = 0; i < a.size(); ++i) s += a.getBit(i) ? '1' : '0';
    return s;
}

int main() {
    const std::size_t p = 13;
    const std::size_t n = 4;                 // force n=4
    // widths c_i: distribute p across n limbs
    std::vector<std::uint64_t> width(n), lo(n);
    for (std::size_t i = 0; i < n; ++i) {
        const std::uint64_t lo128 = (std::uint64_t)i * p / n;
        const std::uint64_t hi128 = (std::uint64_t)(i + 1) * p / n;
        lo[i] = lo128;
        width[i] = hi128 - lo128;
    }
    // digits of s=1000000000101b? s=5 = "101" padded to 13 bits: "0000000000101"
    std::string bits(13, '0');
    bits[12] = '1'; bits[11] = '0'; bits[10] = '1';   // = 5
    ByteArray s = ByteArray::fromString(bits);

    std::vector<std::uint64_t> d(n, 0);
    for (std::size_t i = 0; i < n; ++i)
        for (std::uint64_t b = 0; b < width[i]; ++b)
            if (lo[i] + b < s.size() && s.getBit(s.size() - 1 - (lo[i] + b)))
                d[i] |= std::uint64_t(1) << b;

    std::vector<double> w(n);
    for (std::size_t i = 0; i < n; ++i) {
        const std::uint64_t ipm = (std::uint64_t)i * p % n;
        w[i] = std::pow(2.0, (double)ipm / (double)n);
    }

    // Print the simple digit decomposition: sum d_i * 2^lo[i] should equal s.
    std::uint64_t check = 0;
    for (std::size_t i = 0; i < n; ++i) check += d[i] * (std::uint64_t(1) << lo[i]);
    std::cout << "digit sum = " << check << " (expected 5)\n";
    for (std::size_t i = 0; i < n; ++i)
        std::cout << "  limb " << i << ": lo=" << lo[i] << " w=" << width[i]
                  << " d=" << d[i] << " weight=" << w[i] << "\n";
    return 0;
}
