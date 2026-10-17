#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <string>

#include "bytearray.h"

// Lucas-Lehmer test for Mersenne primes, done entirely on bit / digit arrays:
// no value ever exceeds p bits and no big integer is ever materialized.
//
//   ./mp <start_p> <end_p>
//
// For every odd prime p in [start_p, end_p], decides whether 2^p - 1 is prime
// and prints the ones that are. Large runs show progress on stderr and save a
// checkpoint (ll_<p>.ckpt) every 10000 iterations; just rerun the same command
// after an interruption and it resumes. If the FFT's rounding error ever gets
// too large, the test is retried automatically with a longer transform.
int main(int argc, char** argv) {
    if (argc < 3) {
        std::cerr << "usage: " << argv[0] << " <start_p> <end_p>\n"
                  << "  For each odd prime p in [start_p, end_p], runs the\n"
                  << "  Lucas-Lehmer test to decide if 2^p - 1 is a Mersenne prime.\n";
        return 1;
    }

    std::uint64_t start = 0, end = 0;
    try {
        start = std::stoull(argv[1]);
        end   = std::stoull(argv[2]);
    } catch (const std::exception&) {
        std::cerr << "start_p and end_p must be non-negative integers\n";
        return 1;
    }
    if (start < 3) start = 3;
    if (end < start) {
        std::cerr << "end_p must be >= start_p\n";
        return 1;
    }

    for (std::uint64_t p = start | 1; p <= end; p += 2) {
        if (!ByteArray::isPrime(p)) continue;              // p itself must be prime

        // std::cout << "Checking p=" << p  << std::endl;

        const bool big = p >= 20000;                       // long enough to want progress/checkpoints
        LLOptions opt;
        if (big) {
            opt.progressEvery = std::max<std::uint64_t>(1000, (p - 2) / 100);
            opt.onProgress = [p](std::uint64_t it, std::uint64_t total, double err) {
                std::cerr << "\rp = " << p << "   " << (100 * it / total)
                          << "%   max roundoff " << err << "      " << std::flush;
            };
            opt.checkpointFile = "ll_" + std::to_string(p) + ".ckpt";
            opt.checkpointEvery = 10000;
        }

        LLResult r;
        bool done = false;
        for (std::size_t boost = 0; boost <= 3 && !done; ++boost) {
            opt.fftBoost = boost;
            try {
                r = ByteArray::lucasLehmer(p, opt);
                done = true;
            } catch (const RoundoffError& e) {
                std::cerr << "\n" << e.what() << "\n  -> retrying with a longer FFT"
                          << " (resumes from the last checkpoint if there is one)\n";
            }
        }
        if (big) std::cerr << "\n";
        if (!done) {
            std::cerr << "p = " << p << ": giving up (FFT roundoff too large even with a longer transform)\n";
            continue;
        }
        if (r.resumedFrom)
            std::cerr << "p = " << p << ": resumed from iteration " << r.resumedFrom << "\n";

        if (r.isPrime)
            std::cout << "2^" << p << "-1 is prime!  (Mersenne prime p = " << p << ")" << std::endl;
    }
    return 0;
}