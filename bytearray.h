#ifndef BYTEARRAY_H
#define BYTEARRAY_H

#include <cstddef>
#include <cstdint>
#include <functional>
#include <stdexcept>
#include <string>
#include <vector>

// Precomputed layout for the weighted (Mersenne) transform; defined in the .cpp.
struct MersennePlan;

// Thrown when the FFT's floating-point rounding error gets too large to trust.
struct RoundoffError : std::runtime_error {
    using std::runtime_error::runtime_error;
};

// Options for ByteArray::lucasLehmer.
struct LLOptions {
    // Extra doublings of the FFT length (0 = automatic). Raise this if a run
    // reports RoundoffError; checkpoints stay valid across different values.
    std::size_t fftBoost = 0;
    // Largest allowed distance from an integer before a RoundoffError is thrown.
    double maxAllowedRoundoff = 0.4;
    // Call onProgress(iteration, totalIterations, maxRoundoffSoFar) every
    // progressEvery iterations (0 = never).
    std::uint64_t progressEvery = 0;
    std::function<void(std::uint64_t, std::uint64_t, double)> onProgress;
    // If checkpointFile is set, the state is saved there every checkpointEvery
    // iterations (0 = never) and a matching file is resumed from on startup.
    std::string checkpointFile;
    std::uint64_t checkpointEvery = 0;
};

struct LLResult {
    bool isPrime = false;
    double maxRoundoff = 0.0;          // worst FFT rounding error seen
    std::uint64_t resumedFrom = 0;     // iteration restored from a checkpoint
};

// A space-efficient bit array. Each stored value is packed 8 bits per byte
// (so no more than p/8 bytes hold a p-bit number), never materialized as a
// native integer. The bits are read most-significant-first: position 0 is the
// most significant bit and the last position is the least significant bit,
// i.e. the array reads like a normal written binary number.
class ByteArray {
public:
    void append(std::uint8_t bit);

    // Appends a 1 to the end of the array and prints the new length.
    void appendOne();

    std::size_t size() const;

    // Returns the bit at position i (0 = most significant bit).
    bool getBit(std::size_t i) const;

    // Prints up to maxBits bits (from the most significant side).
    void print(std::size_t maxBits = 64) const;

    // Approximate square root using the binary digit-by-digit method.
    // Works directly on the bits (compare / subtract / shift) and never
    // turns the array into a number. It computes the top `precisionBits`
    // bits of the root correctly, then fills the rest with zeros. Pass a
    // huge value for an exact floor(sqrt). Cost is roughly O(precisionBits^2),
    // no matter how long the array is.
    ByteArray approxSqrt(std::size_t precisionBits = 64) const;

    // Builds an array from a string like "10110" (leftmost char = most significant bit).
    static ByteArray fromString(const std::string& s);

    bool isZero() const;

    // Returns (this mod d) as a bit array. Both the modulus and the result are
    // bit arrays, so no value is ever stored in an integer.
    ByteArray remainder(const ByteArray& d) const;

    bool isDivisibleBy(const ByteArray& d) const;

    // Returns this + other. Both are non-negative bit arrays.
    ByteArray add(const ByteArray& other) const;

    // Returns this - other. Throws std::invalid_argument if this < other.
    ByteArray minus(const ByteArray& other) const;

    // Returns this * other. Uses an FFT-based multiplication on base-2^c limbs
    // for large operands, falling back to schoolbook shift-and-add for small
    // ones. Both paths produce an identical exact result.
    ByteArray multiply(const ByteArray& other) const;

    // Returns this mod (2^p - 1), a value in [0, 2^p - 1], by repeatedly
    // folding the high bits down (2^p ≡ 1 mod (2^p - 1)).
    ByteArray modMersenne(std::size_t p) const;

    // Returns (this * this) mod (2^p - 1) using a weighted (Mersenne) DWT.
    // Inputs longer than p bits are reduced first. Throws RoundoffError if the
    // FFT's rounding error is too large to trust. For long chains of squarings
    // (e.g. Lucas-Lehmer) use lucasLehmer(), which skips all bit conversions.
    ByteArray squareModMersenne(std::size_t p) const;

    // Lucas-Lehmer test for 2^p - 1 (p should be an odd prime). The state is
    // kept as weighted digit arrays between iterations, with the "-2" folded
    // into the carry pass, so no bit-array conversion happens in the loop.
    // Throws RoundoffError if the FFT error exceeds opt.maxAllowedRoundoff.
    static LLResult lucasLehmer(std::uint64_t p, const LLOptions& opt = LLOptions());

    // True if this array's value is greater than or equal to other's.
    bool greaterEqual(const ByteArray& other) const;

    // Adds 2 to the bit array in place (grows the array if needed).
    // Returns *this so calls can be chained.
    ByteArray& addTwo();

    // True if this array's value is strictly larger than other's.
    bool isLargerThan(const ByteArray& other) const;

    // Primality check for a regular number (no bit arrays involved).
    static bool isPrime(std::uint64_t n);

private:
    // Position i (0 = most significant bit) is stored at byte i/8, bit 7-(i%8).
    bool bit(std::size_t i) const {
        return (data_[i / 8] >> (7 - (i % 8))) & 1u;
    }
    void setBit(std::size_t i, bool v) {
        std::uint8_t m = std::uint8_t(1u << (7 - (i % 8)));
        if (v) data_[i / 8] |= m; else data_[i / 8] &= std::uint8_t(~m);
    }
    // Reads len (<= 64) bits starting at MSB-first position start, as a number.
    std::uint64_t readBits(std::size_t start, std::size_t len) const;
    // ORs the low len bits of v into positions [start, start+len) (must be zero there).
    void writeBits(std::size_t start, std::size_t len, std::uint64_t v);
    // -1, 0, +1 for this <, ==, > other (leading zeros ignored).
    int compare(const ByteArray& other) const;
    // Inserts a leading bit at the most-significant side (grows the array by 1).
    void prependBit(bool v);
    // Sets this to the single-bit value zero.
    void setZero();
    // Removes leading (most-significant) zeros so size() is the true bit length.
    void stripLeadingZeros();
    // Base-2^c little-endian limb decomposition of this array.
    std::vector<std::uint64_t> toLimbs(std::size_t c) const;
    // Builds an array from a base-2^c little-endian limb vector.
    static ByteArray fromLimbs(const std::vector<std::uint64_t>& limbs, std::size_t c);
    // Weighted-digit conversions used by the Mersenne transform.
    void toDigits(const MersennePlan& P, std::vector<long long>& g) const;
    static ByteArray fromDigits(const MersennePlan& P, const std::vector<long long>& g);
    // Shift-and-add multiplication; the FFT path's fallback for small operands.
    ByteArray multiplySchoolbook(const ByteArray& other) const;

    std::vector<std::uint8_t> data_;  // packed bytes
    std::size_t size_ = 0;            // number of bits
};

#endif  // BYTEARRAY_H