#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>

#include "bytearray.h"

// Lucas-Lehmer test for Mersenne prime M_p = 2^p - 1 (p odd prime):
//   s0 = 4;  s_i = (s_{i-1}^2 - 2) mod M_p for i = 1..p-2;
//   M_p is prime iff s_{p-2} == 0.
// Everything (squaring, modular reduction, subtract) is done on bit arrays,
// so no value ever exceeds 2p bits and no big integer is ever materialized.
bool isMersennePrime(std::uint64_t p) {
    ByteArray M;                                  // M_p = 2^p - 1 = p ones
    for (std::size_t i = 0; i < p; ++i) M.append(1);

    const ByteArray two = ByteArray::fromString("10");  // 2
    const ByteArray one = ByteArray::fromString("1");   // 1

    ByteArray s = ByteArray::fromString("100");   // s0 = 4
    for (std::size_t i = 0; i + 2 < p; ++i) {     // p-2 squarings
        ByteArray t = s.multiply(s);
        t = t.modMersenne(p);                     // s^2 mod M_p
        if (t.greaterEqual(two)) {
            t = t.minus(two);                     // -2
        } else {                                  // wrap to the top of the range
            t = M.minus(t.isZero() ? two : one);
        }
        s = t;
    }
    return s.isZero();
}

int main(int argc, char** argv) {
    // ./mp <start_p> <end_p> : test every odd prime p in [start_p, end_p]
    // for which 2^p - 1 is a Mersenne prime, using the Lucas-Lehmer test
    // on bit arrays (defined in the ByteArray library).
    if (argc < 3) {
        std::cerr << "usage: " << argv[0] << " <start_p> <end_p>\n"
                  << "  For each odd prime p in [start_p, end_p], runs the\n"
                  << "  Lucas-Lehmer test on bit arrays to decide if 2^p - 1 is\n"
                  << "  a Mersenne prime.\n";
        return 1;
    }
    std::uint64_t start = std::stoull(argv[1]);
    std::uint64_t end   = std::stoull(argv[2]);
    if (start < 3) start = 3;
    if (end < start) {
        std::cerr << "end_p must be >= start_p\n";
        return 1;
    }

    for (std::uint64_t p = start | 1; p <= end; p += 2) {
        if (!ByteArray::isPrime(p)) continue;      // p itself must be prime

        if (isMersennePrime(p))
            std::cout << "2^" << p << "-1 is prime!  (Mersenne prime p = " << p << ")\n";
        //else
        //    std::cout << "p = " << p << " : composite\n";
    }
    return 0;
}
