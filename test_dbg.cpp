// Focused debug: p=61 trial=12 DWT mismatch.
#include <cstdint>
#include <iostream>
#include <string>

#include "bytearray.h"

static std::string toStr(const ByteArray& a) {
    std::string s;
    for (std::size_t i = 0; i < a.size(); ++i) s += a.getBit(i) ? '1' : '0';
    return s;
}

static ByteArray residue(std::size_t p, std::uint64_t seed) {
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

int main() {
    ByteArray s = residue(61, 9001 + 61 * 31 + 12);
    ByteArray got = s.squareModMersenne(61);
    ByteArray want = s.multiply(s).modMersenne(61);
    std::cout << "got  bits=" << got.size() << "\n" << toStr(got) << "\n";
    std::cout << "want bits=" << want.size() << "\n" << toStr(want) << "\n";
    // difference magnitude in bits
    ByteArray a = got, b = want;
    bool aBig = a.isLargerThan(b);
    ByteArray diff = aBig ? a.minus(b) : b.minus(a);
    std::cout << "diff bits=" << diff.size() << "\n" << toStr(diff) << "\n";
    // s^2 without mod
    std::cout << "sq(mod) bits=" << s.multiply(s).modMersenne(61).size() << "\n";
    return 0;
}
