#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

// Little-endian helper arrays used inside the sqrt (index 0 = least significant bit).
using Bits = std::vector<std::uint8_t>;

namespace {

void trim(Bits& v) {                       // drop high-order zeros
    while (!v.empty() && v.back() == 0) v.pop_back();
}

bool geq(const Bits& a, const Bits& b) {   // a >= b (both trimmed)
    if (a.size() != b.size()) return a.size() > b.size();
    for (std::size_t i = a.size(); i-- > 0;)
        if (a[i] != b[i]) return a[i] > b[i];
    return true;
}

void subtract(Bits& a, const Bits& b) {    // a -= b (requires a >= b)
    int borrow = 0;
    for (std::size_t i = 0; i < a.size(); ++i) {
        int d = a[i] - borrow - (i < b.size() ? b[i] : 0);
        if (d < 0) { d += 2; borrow = 1; } else borrow = 0;
        a[i] = static_cast<std::uint8_t>(d);
    }
    trim(a);
}

}  // namespace

// data_[0] is the most significant bit, data_.back() is the least significant
// bit (i.e. the array reads like a normal written binary number).
class ByteArray {
public:
    void append(std::uint8_t bit) { data_.push_back(bit); }

    // Appends a 1 to the end of the array and prints the new length.
    void appendOne() {
        append(1);
        std::cout << "Length: " << data_.size() << std::endl;
    }

    std::size_t size() const { return data_.size(); }

    // Prints up to maxBits bits (from the most significant side).
    void print(std::size_t maxBits = 64) const {
        std::string s;
        for (std::size_t i = 0; i < data_.size() && i < maxBits; ++i)
            s += data_[i] ? '1' : '0';
        if (data_.size() > maxBits) s += "...";
        std::cout << s << "  (" << data_.size() << " bits)" << std::endl;
    }

    // Approximate square root using the binary digit-by-digit method.
    // Works directly on the bits (compare / subtract / shift) and never
    // turns the array into a number. It computes the top `precisionBits`
    // bits of the root correctly, then fills the rest with zeros. Pass a
    // huge value for an exact floor(sqrt). Cost is roughly O(precisionBits^2),
    // no matter how long the array is.
    ByteArray approxSqrt(std::size_t precisionBits = 64) const {
        ByteArray result;
        const std::size_t n = data_.size();
        if (n == 0) return result;

        const std::size_t totalPairs = (n + 1) / 2;
        const std::size_t pairs = std::min(precisionBits, totalPairs);

        Bits R, Q;                 // remainder and root so far (little-endian)
        std::size_t idx = 0;
        for (std::size_t p = 0; p < pairs; ++p) {
            std::uint8_t hi, lo;
            if (p == 0 && n % 2 == 1) { hi = 0; lo = data_[0]; idx = 1; }
            else { hi = data_[idx]; lo = data_[idx + 1]; idx += 2; }

            // R = R*4 + (next two input bits)
            R.insert(R.begin(), {lo, hi});
            trim(R);

            // T = Q*4 + 1
            Bits T{1, 0};
            T.insert(T.end(), Q.begin(), Q.end());
            trim(T);

            // Next root bit is 1 if R >= T
            std::uint8_t bit = 0;
            if (geq(R, T)) { subtract(R, T); bit = 1; }
            Q.insert(Q.begin(), bit);    // Q = Q*2 + bit
        }

        result.data_.assign(Q.rbegin(), Q.rend());                 // MSB first
        result.data_.resize(result.data_.size() + (totalPairs - pairs), 0);
        auto first = std::find(result.data_.begin(), result.data_.end(), 1);
        result.data_.erase(result.data_.begin(), first);           // strip leading zeros
        if (result.data_.empty()) result.data_.push_back(0);
        return result;
    }

    // Builds an array from a string like "10110" (leftmost char = most significant bit).
    static ByteArray fromString(const std::string& s) {
        ByteArray r;
        for (char c : s) {
            if (c != '0' && c != '1') throw std::invalid_argument("only '0' and '1' allowed");
            r.append(c == '1' ? 1 : 0);
        }
        return r;
    }

    bool isZero() const {
        return std::none_of(data_.begin(), data_.end(), [](std::uint8_t b) { return b != 0; });
    }

    // Returns (this mod d) as a bit array. Both the modulus and the result are
    // bit arrays, so no value is ever stored in an integer.
    // Reads this array from most to least significant bit and keeps a running
    // remainder R (a small bit array, never longer than d):
    //     R = 2*R + bit;  if (R >= d) R -= d;
    // Cost is roughly O(length of this array * length of d). Requires d != 0.
    ByteArray remainder(const ByteArray& d) const {
        Bits D(d.data_.rbegin(), d.data_.rend());      // to little-endian
        trim(D);
        if (D.empty()) throw std::invalid_argument("modulus must be > 0");

        Bits R;
        for (std::uint8_t bit : data_) {
            R.insert(R.begin(), bit);                  // R = 2*R + bit
            trim(R);
            if (geq(R, D)) subtract(R, D);             // R < 2*D here, so once is enough
        }

        ByteArray result;
        result.data_.assign(R.rbegin(), R.rend());     // back to MSB-first
        if (result.data_.empty()) result.data_.push_back(0);
        return result;
    }

    bool isDivisibleBy(const ByteArray& d) const {
        if (d.isZero()) return false;
        return remainder(d).isZero();
    }

    // 1) Adds 2 to the bit array in place (grows the array if needed).
    // Adding 2 is the same as adding 1 at bit position 1, so we start there
    // and carry upward. Returns *this so calls can be chained.
    ByteArray& addTwo() {
        if (data_.size() < 2) data_.insert(data_.begin(), 2 - data_.size(), 0);
        std::size_t i = data_.size() - 2;              // bit position 1
        while (true) {
            if (data_[i] == 0) { data_[i] = 1; break; }
            data_[i] = 0;                              // 1 + 1 = 0, carry
            if (i == 0) { data_.insert(data_.begin(), 1); break; }  // carry out the top
            --i;
        }
        return *this;
    }

    // 2) True if this array's value is strictly larger than other's.
    // Leading zeros are ignored. Compares lengths first, then bit by bit
    // from the most significant side.
    bool isLargerThan(const ByteArray& other) const {
        auto a = std::find(data_.begin(), data_.end(), std::uint8_t{1});
        auto b = std::find(other.data_.begin(), other.data_.end(), std::uint8_t{1});
        const std::size_t lenA = static_cast<std::size_t>(data_.end() - a);
        const std::size_t lenB = static_cast<std::size_t>(other.data_.end() - b);
        if (lenA != lenB) return lenA > lenB;
        for (; a != data_.end(); ++a, ++b)
            if (*a != *b) return *a > *b;
        return false;                                  // equal
    }

    // 3) Primality check for a regular number (no bit arrays involved).
    // Trial division by 2, 3, then 6k +/- 1 up to sqrt(n).
    static bool isPrime(std::uint64_t n) {
        if (n < 2) return false;
        if (n < 4) return true;                        // 2 and 3
        if (n % 2 == 0 || n % 3 == 0) return false;
        for (std::uint64_t i = 5; i <= n / i; i += 6)  // n / i avoids overflow
            if (n % i == 0 || n % (i + 2) == 0) return false;
        return true;
    }

private:
    std::vector<std::uint8_t> data_;
};

int main(int argc, char** argv) {
    // ./mp <start_p> <end_p> : test every odd prime p in [start_p, end_p]
    // for which 2^p - 1 is a Mersenne prime.
    if (argc < 3) {
        std::cerr << "usage: " << argv[0] << " <start_p> <end_p>\n"
                  << "  For each odd prime p in [start_p, end_p], builds the bit\n"
                  << "  array m = 2^p - 1 and trial-divides it by every odd f from\n"
                  << "  3 up to sqrt(m) to decide if it's a Mersenne prime.\n";
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

        // m = 2^p - 1, i.e. a bit array of p ones.
        ByteArray m;
        for (std::size_t i = 0; i < p; ++i) m.append(1);

        // Exact floor(sqrt(m)); has ~p/2 bits.
        ByteArray sqrtM = m.approxSqrt(m.size());

        // Trial divide by every odd f from 3 up to sqrt(m).
        ByteArray f = ByteArray::fromString("11");   // 3
        bool mersenne = true;
        while (!f.isLargerThan(sqrtM)) {
            if (m.isDivisibleBy(f)) { mersenne = false; break; }
            f.addTwo();                              // next odd candidate
        }

        if (mersenne)
            std::cout << "2^" << p << "-1 is prime!  (Mersenne prime p = " << p << ")\n";
        else
            std::cout << "p = " << p << " : composite\n";
    }
    return 0;
}
