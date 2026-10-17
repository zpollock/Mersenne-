#ifndef BYTEARRAY_H
#define BYTEARRAY_H

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

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

    // Returns this - other (requires this >= other).
    ByteArray minus(const ByteArray& other) const;

    // Returns this * other. Uses an FFT-based multiplication on base-2^c limbs
    // for large operands, falling back to schoolbook shift-and-add for small
    // ones. Both paths produce an identical exact result.
    ByteArray multiply(const ByteArray& other) const;

    // Returns this mod (2^p - 1), a value in [0, 2^p - 1], by repeatedly
    // folding the high bits down (2^p ≡ 1 mod (2^p - 1)).
    ByteArray modMersenne(std::size_t p) const;

    // Returns (this * this) mod (2^p - 1). The Mersenne reduction is fused
    // into the transform via a weighted (Mersenne) DWT: the operand is split
    // into n weighted limbs and a single cyclic transform of length n yields
    // the residue directly, so no value ever exceeds p bits and no separate
    // post-square fold is needed. Only meaningful when this < 2^p.
    ByteArray squareModMersenne(std::size_t p) const;

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
    // Shift-and-add multiplication; the FFT path's fallback for small operands.
    ByteArray multiplySchoolbook(const ByteArray& other) const;

    std::vector<std::uint8_t> data_;  // packed bytes
    std::size_t size_ = 0;            // number of bits
};

#endif  // BYTEARRAY_H
