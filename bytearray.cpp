#include "bytearray.h"

#include <algorithm>
#include <cmath>
#include <complex>
#include <stdexcept>

#include <iostream>  // for appendOne() / print()

namespace {

// Little-endian packed bit array used for the working buffers of the
// square-root and remainder algorithms: bit 0 is the least significant bit.
// Everything stays packed 8 bits per byte, so no 8x waste anywhere.
struct Bl {
    std::vector<std::uint8_t> d;   // packed bytes
    std::size_t n = 0;             // number of bits

    bool bit(std::size_t k) const { return (d[k / 8] >> (k % 8)) & 1u; }
    void set(std::size_t k, bool v) {
        std::uint8_t m = std::uint8_t(1u << (k % 8));
        if (v) d[k / 8] |= m; else d[k / 8] &= std::uint8_t(~m);
    }
    void ensure(std::size_t k) { if (d.size() <= k / 8) d.resize(k / 8 + 1, 0); }
    bool empty() const { return n == 0; }
};

void trim(Bl& v) {                     // drop high-order (most significant) zeros
    while (v.n && !v.bit(v.n - 1)) --v.n;
    if (v.n) v.d.resize((v.n + 7) / 8); else v.d.clear();
}

bool geq(const Bl& a, const Bl& b) {   // a >= b
    if (a.n != b.n) return a.n > b.n;
    for (std::size_t i = a.n; i-- > 0;)
        if (a.bit(i) != b.bit(i)) return a.bit(i) > b.bit(i);
    return true;
}

void subtract(Bl& a, const Bl& b) {    // a -= b (requires a >= b)
    int borrow = 0;
    for (std::size_t i = 0; i < a.n; ++i) {
        int d = (a.bit(i) ? 1 : 0) - borrow - ((i < b.n && b.bit(i)) ? 1 : 0);
        if (d < 0) { d += 2; borrow = 1; } else borrow = 0;
        a.set(i, d != 0);
    }
    trim(a);
}

// R = R*4 + hi*2 + lo   (prepend two high bits: shift left 2, then OR them in)
void shl2Add(Bl& R, bool hi, bool lo) {
    std::size_t on = R.n;
    R.n += 2;
    R.ensure(R.n - 1);
    for (std::size_t k = on; k-- > 0;) R.set(k + 2, R.bit(k));
    R.set(0, lo);
    R.set(1, hi);
}

// Q = Q*2 + bit   (shift left 1, then OR the new low bit)
void shl1Add(Bl& Q, bool bit) {
    std::size_t on = Q.n;
    Q.n += 1;
    Q.ensure(Q.n - 1);
    for (std::size_t k = on; k-- > 0;) Q.set(k + 1, Q.bit(k));
    Q.set(0, bit);
}

// T = Q*4 + 1
Bl qmul4plus1(const Bl& Q) {
    Bl T;
    T.n = Q.n + 2;
    T.ensure(T.n - 1);
    T.set(0, 1);
    T.set(1, 0);
    for (std::size_t k = 0; k < Q.n; ++k) T.set(k + 2, Q.bit(k));
    return T;
}

// Iterative radix-2 Cooley-Tukey FFT on std::complex<double>.
// Converts a vector of limbs to/from the frequency domain. The inverse pass
// (invert = true) also divides by N. Used for convolution-based multiplication.
void fft(std::vector<std::complex<double>>& a, bool invert) {
    const std::size_t n = a.size();
    if (n < 2) return;

    // Bit-reversal permutation.
    for (std::size_t i = 1, j = 0; i < n; ++i) {
        std::size_t bit = n >> 1;
        for (; j & bit; bit >>= 1) j ^= bit;
        j ^= bit;
        if (i < j) std::swap(a[i], a[j]);
    }

    // Roots computed directly (not by repeated multiplication), cached per n.
    // sin/cos in long double keep the accumulated error tiny even for large
    // transforms and for the many same-size FFTs in a Lucas-Lehmer run.
    static std::size_t cachedN = 0;
    static std::vector<std::complex<double>> roots;   // exp(-2*pi*i*k/n), k < n/2
    if (cachedN != n) {
        roots.assign(n / 2, std::complex<double>(0, 0));
        const long double pi = std::acos(-1.0L);
        for (std::size_t k = 0; k < n / 2; ++k) {
            const long double ang = 2.0L * pi * k / n;
            roots[k] = std::complex<double>(double(std::cos(ang)), double(-std::sin(ang)));
        }
        cachedN = n;
    }

    for (std::size_t len = 2; len <= n; len <<= 1) {
        const std::size_t half = len / 2, step = n / len;
        for (std::size_t i = 0; i < n; i += len) {
            for (std::size_t j = 0; j < half; ++j) {
                std::complex<double> w = roots[j * step];
                if (invert) w = std::conj(w);
                std::complex<double> u = a[i + j];
                std::complex<double> v = a[i + j + half] * w;
                a[i + j] = u + v;
                a[i + j + half] = u - v;
            }
        }
    }

    if (invert)
        for (std::complex<double>& x : a) x /= static_cast<double>(n);
}

}  // namespace

void ByteArray::append(std::uint8_t bit) {
    if (size_ % 8 == 0) data_.push_back(0);   // start a fresh byte
    setBit(size_, bit != 0);
    ++size_;
}

void ByteArray::appendOne() {
    append(1);
    std::cout << "Length: " << size_ << std::endl;
}

std::size_t ByteArray::size() const { return size_; }

bool ByteArray::getBit(std::size_t i) const { return bit(i); }

void ByteArray::print(std::size_t maxBits) const {
    std::string s;
    for (std::size_t i = 0; i < size_ && i < maxBits; ++i)
        s += bit(i) ? '1' : '0';
    if (size_ > maxBits) s += "...";
    std::cout << s << "  (" << size_ << " bits)" << std::endl;
}

ByteArray ByteArray::approxSqrt(std::size_t precisionBits) const {
    ByteArray result;
    if (size_ == 0) return result;

    const std::size_t totalPairs = (size_ + 1) / 2;
    const std::size_t pairs = std::min(precisionBits, totalPairs);

    Bl R, Q;                 // remainder and root so far (little-endian)
    std::size_t idx = 0;
    for (std::size_t p = 0; p < pairs; ++p) {
        bool hi, lo;
        if (p == 0 && size_ % 2 == 1) { hi = false; lo = bit(0); idx = 1; }
        else { hi = bit(idx); lo = bit(idx + 1); idx += 2; }

        shl2Add(R, hi, lo);  // R = R*4 + hi*2 + lo
        trim(R);

        Bl T = qmul4plus1(Q);  // T = Q*4 + 1
        trim(T);

        bool b = false;
        if (geq(R, T)) { subtract(R, T); b = true; }
        shl1Add(Q, b);      // Q = Q*2 + b
    }

    // Emit Q (little-endian root) as an MSB-first array of width totalPairs,
    // then strip any leading zeros (keep at least "0").
    result.size_ = totalPairs;
    result.data_.assign((totalPairs + 7) / 8, 0);
    for (std::size_t j = 0; j < Q.n; ++j)
        result.setBit(totalPairs - 1 - j, Q.bit(j));

    std::size_t first = totalPairs;
    for (std::size_t i = 0; i < totalPairs; ++i)
        if (result.bit(i)) { first = i; break; }
    if (first == totalPairs) { result.setZero(); return result; }
    if (first > 0) {
        std::size_t nw = totalPairs - first;
        ByteArray r2;
        r2.size_ = nw;
        r2.data_.assign((nw + 7) / 8, 0);
        for (std::size_t i = 0; i < nw; ++i)
            r2.setBit(i, result.bit(first + i));
        result = r2;
    }
    return result;
}

ByteArray ByteArray::fromString(const std::string& s) {
    ByteArray r;
    for (char c : s) {
        if (c != '0' && c != '1') throw std::invalid_argument("only '0' and '1' allowed");
        r.append(c == '1' ? 1 : 0);
    }
    return r;
}

bool ByteArray::isZero() const {
    for (std::size_t i = 0; i < size_; ++i)
        if (bit(i)) return false;
    return true;
}

ByteArray ByteArray::remainder(const ByteArray& d) const {
    // Little-endian copy of the divisor (d is MSB-first).
    Bl D;
    D.n = d.size_;
    if (D.n) {
        D.ensure(D.n - 1);
        for (std::size_t j = 0; j < D.n; ++j) D.set(j, d.bit(D.n - 1 - j));
    }
    trim(D);
    if (D.empty()) throw std::invalid_argument("modulus must be > 0");

    // Read this array MSB-first, keeping a running remainder R: R = 2*R + bit;
    // if (R >= d) R -= d.
    Bl R;
    for (std::size_t k = 0; k < size_; ++k) {
        shl1Add(R, bit(k));
        trim(R);
        if (geq(R, D)) subtract(R, D);
    }

    ByteArray result;
    if (R.n == 0) { result.setZero(); return result; }
    result.size_ = R.n;
    result.data_.assign((R.n + 7) / 8, 0);
    for (std::size_t j = 0; j < R.n; ++j)
        result.setBit(R.n - 1 - j, R.bit(j));
    return result;
}

bool ByteArray::isDivisibleBy(const ByteArray& d) const {
    if (d.isZero()) return false;
    return remainder(d).isZero();
}

ByteArray ByteArray::add(const ByteArray& other) const {
    std::size_t n = std::max(size_, other.size_);
    ByteArray r;
    r.size_ = n + 1;
    r.data_.assign((n + 1 + 7) / 8, 0);

    int carry = 0;
    for (std::size_t i = 0; i <= n; ++i) {
        int a = (i < size_ && bit(size_ - 1 - i)) ? 1 : 0;   // 2^i place of this
        int b = (i < other.size_ && other.bit(other.size_ - 1 - i)) ? 1 : 0;
        int s = a + b + carry;
        r.setBit(n - i, s & 1);
        carry = s >> 1;
    }
    r.stripLeadingZeros();
    return r;
}

ByteArray ByteArray::minus(const ByteArray& other) const {
    // Requires *this >= other.
    int borrow = 0;
    ByteArray r;
    r.size_ = size_;
    r.data_.assign((size_ + 7) / 8, 0);
    for (std::size_t i = 0; i < size_; ++i) {
        int a = bit(size_ - 1 - i) ? 1 : 0;
        int b = (i < other.size_ && other.bit(other.size_ - 1 - i)) ? 1 : 0;
        int d = a - b - borrow;
        if (d < 0) { d += 2; borrow = 1; } else borrow = 0;
        r.setBit(size_ - 1 - i, d != 0);
    }
    r.stripLeadingZeros();
    return r;
}

ByteArray ByteArray::multiply(const ByteArray& other) const {
    if (isZero() || other.isZero()) {
        ByteArray z;
        z.setZero();
        return z;
    }

    // The FFT path only wins on large operands; schoolbook is cheaper and
    // identical in result for small ones, so switch on a bit-length threshold.
    const std::size_t maxBits = std::max(size_, other.size_);
    if (maxBits < 1024) return multiplySchoolbook(other);

    // Choose chunk size c so no convolution coefficient over a double's exact
    // integer range. The largest coefficient is ~n * (2^c)^2 (n limbs), so we
    // need log2(n) + 2c comfortably below 53; use 45 for FFT error headroom.
    std::size_t c = 30;
    while (c > 1) {
        const std::size_t n = (maxBits + c - 1) / c;
        const long double lb = std::log2(static_cast<long double>(n))
                             + 2.0L * static_cast<long double>(c);
        if (lb < 45.0L) break;
        --c;
    }

    std::vector<std::uint64_t> A = toLimbs(c);
    std::vector<std::uint64_t> B = other.toLimbs(c);

    std::size_t need = A.size() + B.size() - 1;
    std::size_t N = 1;
    while (N < need) N <<= 1;

    std::vector<std::complex<double>> FA(N), FB(N);
    for (std::size_t i = 0; i < A.size(); ++i)
        FA[i] = std::complex<double>(static_cast<double>(A[i]), 0.0);
    for (std::size_t i = 0; i < B.size(); ++i)
        FB[i] = std::complex<double>(static_cast<double>(B[i]), 0.0);

    fft(FA, false);
    fft(FB, false);
    for (std::size_t i = 0; i < N; ++i) FA[i] *= FB[i];
    fft(FA, true);

    // Round the (near-integer) coefficients and propagate the base-2^c carry.
    const std::uint64_t mask = (std::uint64_t(1) << c) - 1;
    std::vector<std::uint64_t> prod;
    prod.reserve(need + 1);
    std::uint64_t carry = 0;
    for (std::size_t i = 0; i < need; ++i) {
        const long long r = std::llround(FA[i].real());
        // A tiny negative FFT residue could round below zero; clamp so the
        // cast to uint64_t doesn't wrap around to a huge value.
        const std::uint64_t coeff = r > 0
                                      ? static_cast<std::uint64_t>(r)
                                      : std::uint64_t(0);
        const std::uint64_t total = coeff + carry;
        prod.push_back(total & mask);
        carry = total >> c;
    }
    while (carry) {
        prod.push_back(carry & mask);
        carry >>= c;
    }

    return fromLimbs(prod, c);
}

ByteArray ByteArray::multiplySchoolbook(const ByteArray& other) const {
    ByteArray acc;
    acc.setZero();
    for (std::size_t j = 0; j < other.size_; ++j) {
        if (!other.bit(other.size_ - 1 - j)) continue;   // 2^j place of other
        ByteArray shifted = *this;
        for (std::size_t z = 0; z < j; ++z) shifted.append(0);  // *this << j
        acc = acc.add(shifted);
    }
    return acc;
}

std::vector<std::uint64_t> ByteArray::toLimbs(std::size_t c) const {
    const std::size_t n = (size_ + c - 1) / c;
    std::vector<std::uint64_t> v(n, 0);
    for (std::size_t i = 0; i < size_; ++i) {
        if (!bit(i)) continue;
        const std::size_t g = size_ - 1 - i;      // little-endian bit index
        const std::size_t limb = g / c, off = g % c;
        v[limb] |= std::uint64_t(1) << off;
    }
    return v;
}

ByteArray ByteArray::fromLimbs(const std::vector<std::uint64_t>& limbs,
                               std::size_t c) {
    // Highest set bit across all limbs: limb i covers global bits [i*c, i*c+c).
    std::size_t highest = 0;   // highest global bit index
    for (std::size_t i = 0; i < limbs.size(); ++i) {
        std::uint64_t v = limbs[i];
        if (!v) continue;
        std::size_t h = 0;
        while (v >>= 1) ++h;
        highest = std::max(highest, i * c + h);
    }

    ByteArray r;
    r.size_ = highest + 1;
    r.data_.assign((r.size_ + 7) / 8, 0);
    for (std::size_t i = 0; i < limbs.size(); ++i)
        for (std::size_t b = 0; b < c; ++b)
            if ((limbs[i] >> b) & 1u)
                r.setBit(r.size_ - 1 - (i * c + b), true);
    return r;
}

ByteArray ByteArray::modMersenne(std::size_t p) const {
    ByteArray x = *this;
    while (x.size_ > p) {
        const std::size_t hiLen = x.size_ - p;
        ByteArray hi, lo;
        hi.size_ = hiLen;
        hi.data_.assign((hiLen + 7) / 8, 0);
        for (std::size_t i = 0; i < hiLen; ++i) hi.setBit(i, x.bit(i));
        lo.size_ = p;
        lo.data_.assign((p + 7) / 8, 0);
        for (std::size_t i = 0; i < p; ++i) lo.setBit(i, x.bit(hiLen + i));
        x = hi.add(lo);                 // 2^p ≡ 1 (mod 2^p - 1)
    }

    // x <= 2^p - 1 now. The only value >= M is M itself (all ones) == 0 (mod M).
    if (x.size_ == p) {
        bool allOnes = true;
        for (std::size_t i = 0; i < p; ++i)
            if (!x.bit(i)) { allOnes = false; break; }
        if (allOnes) return ByteArray::fromString("0");
    }
    return x;
}

ByteArray ByteArray::squareModMersenne(std::size_t p) const {
    // Weighted (Mersenne) DWT: split this (< 2^p) into n weighted limbs and do
    // a single cyclic transform of length n, so the result is s^2 mod (2^p-1)
    // with no separate post-square fold. n is a power of two.
    //
    // Use ceil limb positions lo[i] = ceil(i*p/n) (with lo[0]=0, lo[n]=p).
    // Along with weights w[i] = 2^(lo[i] - i*p/n) >= 1 applied forward and
    // divided out on the way back, the product terms come out as exact
    // integers (floor positions would make them half-integers and corrupt the
    // rounding).
    std::size_t n = 1;
    for (;;) {
        const std::size_t maxc = (p + n - 1) / n;          // ceil(p/n)
        const long double lb = std::log2(static_cast<long double>(n))
                             + 2.0L * static_cast<long double>(maxc);
        if (n >= 2 && lb < 50.0L) break;                   // exact in double
        n <<= 1;
    }

    // Split into weighted digits.
    std::vector<std::uint64_t> d(n, 0);
    std::vector<std::uint64_t> width(n);
    std::vector<std::uint64_t> lo(n);
    std::vector<double> w(n);
    for (std::size_t i = 0; i < n; ++i) {
        const unsigned __int128 lo128 = ((unsigned __int128)i * p + n - 1) / n;  // ceil
        const unsigned __int128 hi128 = ((unsigned __int128)(i + 1) * p + n - 1) / n;
        lo[i] = static_cast<std::uint64_t>(lo128);
        width[i] = static_cast<std::uint64_t>(hi128 - lo128);

        std::uint64_t v = 0;
        for (std::uint64_t b = 0; b < width[i]; ++b) {
            if (lo[i] + b < size_ && bit(size_ - 1 - (lo[i] + b)))
                v |= std::uint64_t(1) << b;
        }
        d[i] = v;

        const unsigned __int128 rem = ((unsigned __int128)i * p) % n;
        const unsigned __int128 k = (n - rem) % n;         // lo[i]*n - i*p <= n-1
        w[i] = std::pow(2.0, static_cast<double>(k) / static_cast<double>(n));
    }

    // One forward transform, pointwise square, one inverse transform.
    std::vector<std::complex<double>> A(n);
    for (std::size_t i = 0; i < n; ++i)
        A[i] = std::complex<double>(static_cast<double>(d[i]) * w[i], 0.0);
    fft(A, false);
    for (std::size_t i = 0; i < n; ++i) A[i] = A[i] * A[i];
    fft(A, true);

    // Divide out the weights, round to integers, then propagate carries
    // cyclically (the carry that leaves the top limb wraps back onto limb 0,
    // which IS the mod-(2^p-1) reduction).
    std::vector<std::uint64_t> g(n);
    for (std::size_t i = 0; i < n; ++i)
        g[i] = static_cast<std::uint64_t>(std::llround(A[i].real() / w[i]));

    for (int pass = 0; pass < 16; ++pass) {
        std::uint64_t carry = 0;
        for (std::size_t i = 0; i < n; ++i) {
            const std::uint64_t base = std::uint64_t(1) << width[i];
            const std::uint64_t v = g[i] + carry;
            g[i] = v % base;
            carry = v / base;
        }
        if (carry == 0) break;
        g[0] += carry;                 // cyclic wrap: feed it back into limb 0
    }

    // Reassemble the residue (value = sum g_i * 2^lo[i]) as an MSB-first array.
    std::size_t highest = 0;
    for (std::size_t i = 0; i < n; ++i) {
        std::uint64_t v = g[i];
        if (!v) continue;
        std::size_t h = 0;
        while (v >>= 1) ++h;
        if (lo[i] + h > highest) highest = lo[i] + h;
    }

    ByteArray r;
    r.size_ = highest + 1;
    r.data_.assign((r.size_ + 7) / 8, 0);
    for (std::size_t i = 0; i < n; ++i)
        for (std::uint64_t b = 0; b < width[i]; ++b)
            if ((g[i] >> b) & 1u)
                r.setBit(r.size_ - 1 - (lo[i] + b), true);

    // A canonical Mersenne residue never equals M; if it came out all ones
    // (value == 2^p - 1) the true residue is 0.
    if (r.size_ == p + 1) {
        r.stripLeadingZeros();
    }
    if (r.size() == p) {
        bool allOnes = true;
        for (std::size_t i = 0; i < p; ++i)
            if (!r.bit(i)) { allOnes = false; break; }
        if (allOnes) return ByteArray::fromString("0");
    }
    return r;
}

bool ByteArray::greaterEqual(const ByteArray& other) const {
    std::size_t lenA = 0, aTop = 0, lenB = 0, bTop = 0;
    for (std::size_t i = 0; i < size_; ++i)
        if (bit(i)) { lenA = size_ - i; aTop = i; break; }
    for (std::size_t i = 0; i < other.size_; ++i)
        if (other.bit(i)) { lenB = other.size_ - i; bTop = i; break; }

    if (lenA != lenB) return lenA > lenB;
    for (std::size_t k = 0; k < lenA; ++k) {
        bool ba = bit(aTop + k), bb = other.bit(bTop + k);
        if (ba != bb) return ba;
    }
    return true;                        // equal
}

ByteArray& ByteArray::addTwo() {
    if (size_ == 0) {                 // 0 -> 2 = "10"
        data_.resize(1, 0);
        size_ = 2;
        setBit(0, true); setBit(1, false);
        return *this;
    }
    if (size_ == 1) prependBit(false);   // make room for the carry chain

    std::size_t i = size_ - 2;        // bit position 1 = the "2" place
    while (true) {
        if (!bit(i)) { setBit(i, true); break; }
        setBit(i, false);             // 1 + 1 = 0, carry
        if (i == 0) { prependBit(true); break; }  // carry out the top
        --i;
    }
    return *this;
}

bool ByteArray::isLargerThan(const ByteArray& other) const {
    std::size_t lenA = 0, aTop = 0, lenB = 0, bTop = 0;
    for (std::size_t i = 0; i < size_; ++i)
        if (bit(i)) { lenA = size_ - i; aTop = i; break; }
    for (std::size_t i = 0; i < other.size_; ++i)
        if (other.bit(i)) { lenB = other.size_ - i; bTop = i; break; }

    if (lenA != lenB) return lenA > lenB;
    for (std::size_t k = 0; k < lenA; ++k) {
        bool ba = bit(aTop + k), bb = other.bit(bTop + k);
        if (ba != bb) return ba;
    }
    return false;                     // equal
}

bool ByteArray::isPrime(std::uint64_t n) {
    if (n < 2) return false;
    if (n < 4) return true;           // 2 and 3
    if (n % 2 == 0 || n % 3 == 0) return false;
    for (std::uint64_t i = 5; i <= n / i; i += 6)  // n / i avoids overflow
        if (n % i == 0 || n % (i + 2) == 0) return false;
    return true;
}

// Private helpers ------------------------------------------------------------

void ByteArray::prependBit(bool v) {
    data_.resize(data_.size() + 1, 0);
    for (std::size_t j = size_; j-- > 0;) setBit(j + 1, bit(j));
    ++size_;
    setBit(0, v);
}

void ByteArray::setZero() {
    data_.assign(1, 0);
    size_ = 1;
}

void ByteArray::stripLeadingZeros() {
    std::size_t first = size_;
    for (std::size_t i = 0; i < size_; ++i)
        if (bit(i)) { first = i; break; }
    if (first == size_) { setZero(); return; }
    if (first == 0) return;

    const std::size_t nw = size_ - first;
    ByteArray r;
    r.size_ = nw;
    r.data_.assign((nw + 7) / 8, 0);
    for (std::size_t i = 0; i < nw; ++i) r.setBit(i, bit(first + i));
    *this = r;
}
