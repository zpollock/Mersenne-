#include "bytearray.h"

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <stdexcept>

#include <iostream>  // for appendOne() / print()

// ---------------------------------------------------------------------------
// Layout of the weighted (Mersenne) transform for a given exponent p.
// Computed once and reused for every squaring of the same p.
//   n          transform length (power of two)
//   lo[i]      ceil(i*p/n): first bit of digit i (lo[n] = p)
//   width[i]   lo[i+1] - lo[i]: number of bits in digit i
//   w[i]       2^(lo[i] - i*p/n) >= 1   (applied before the FFT)
//   winv[i]    1 / w[i]                 (applied after the inverse FFT)
// ---------------------------------------------------------------------------
struct MersennePlan {
    std::uint64_t p = 0;
    std::size_t boost = 0;
    std::size_t n = 0;
    std::vector<std::size_t> lo;
    std::vector<unsigned> width;
    std::vector<double> w, winv;
};

namespace {

// ----- Little-endian packed bit buffers (square-root and remainder) --------

// Little-endian packed bit array used for the working buffers of the
// square-root and remainder algorithms: bit 0 is the least significant bit.
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

// R = R*4 + hi*2 + lo
void shl2Add(Bl& R, bool hi, bool lo) {
    std::size_t on = R.n;
    R.n += 2;
    R.ensure(R.n - 1);
    for (std::size_t k = on; k-- > 0;) R.set(k + 2, R.bit(k));
    R.set(0, lo);
    R.set(1, hi);
}

// Q = Q*2 + bit
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

// ----- FFT ------------------------------------------------------------------

// Per-size tables for the FFT, cached per thread (a Lucas-Lehmer run uses the
// same size every iteration, so these are built once).
//   rev[i]       bit-reversal permutation
//   twr/twi      twiddles stored contiguously per stage:
//                tw[half + j] = exp(-2*pi*i*j / (2*half)), j < half
// Twiddles are computed directly with long-double sin/cos (not by repeated
// multiplication, which loses accuracy on large transforms).
struct FftTables {
    std::size_t n = 0;
    std::vector<std::uint32_t> rev;
    std::vector<double> twr, twi;
};

void buildFftTables(FftTables& T, std::size_t n) {
    T.n = n;
    T.rev.assign(n, 0);
    for (std::size_t i = 1, j = 0; i < n; ++i) {
        std::size_t bit = n >> 1;
        for (; j & bit; bit >>= 1) j ^= bit;
        j ^= bit;
        T.rev[i] = static_cast<std::uint32_t>(j);
    }
    T.twr.assign(n, 0.0);
    T.twi.assign(n, 0.0);
    const long double pi = std::acos(-1.0L);
    for (std::size_t half = 1; half < n; half <<= 1) {
        for (std::size_t j = 0; j < half; ++j) {
            const long double ang = pi * static_cast<long double>(j) / static_cast<long double>(half);
            T.twr[half + j] = static_cast<double>(std::cos(ang));
            T.twi[half + j] = static_cast<double>(-std::sin(ang));
        }
    }
}

// Iterative radix-2 Cooley-Tukey FFT on std::complex<double>. The inverse
// pass (invert = true) also divides by N. Works on the raw (re, im) doubles
// with contiguous per-stage twiddles; about 6-10x faster than a version that
// strides through one shared twiddle table and uses operator* on std::complex.
void fft(std::vector<std::complex<double>>& a, bool invert) {
    const std::size_t n = a.size();
    if (n < 2) return;

    static thread_local FftTables T;
    if (T.n != n) buildFftTables(T, n);

    // std::complex<double> is guaranteed layout-compatible with double[2].
    double* d = reinterpret_cast<double*>(a.data());

    for (std::size_t i = 1; i < n; ++i) {                  // bit-reversal permutation
        const std::size_t j = T.rev[i];
        if (i < j) { std::swap(d[2 * i], d[2 * j]); std::swap(d[2 * i + 1], d[2 * j + 1]); }
    }

    for (std::size_t i = 0; i < n; i += 2) {               // first stage: twiddle is 1
        double* x = d + 2 * i;
        const double ur = x[0], ui = x[1], vr = x[2], vi = x[3];
        x[0] = ur + vr; x[1] = ui + vi; x[2] = ur - vr; x[3] = ui - vi;
    }

    const double sgn = invert ? -1.0 : 1.0;                // inverse uses the conjugate root
    for (std::size_t half = 2; half < n; half <<= 1) {
        const std::size_t len = half * 2;
        const double* tr = &T.twr[half];
        const double* ti = &T.twi[half];
        for (std::size_t i = 0; i < n; i += len) {
            double* x = d + 2 * i;
            double* y = x + 2 * half;
            for (std::size_t j = 0; j < half; ++j) {
                const double wr = tr[j], wi = sgn * ti[j];
                const double yr = y[2 * j], yi = y[2 * j + 1];
                const double vr = yr * wr - yi * wi;       // y * w, written out
                const double vi = yr * wi + yi * wr;
                const double ur = x[2 * j], ui = x[2 * j + 1];
                x[2 * j] = ur + vr;     x[2 * j + 1] = ui + vi;
                y[2 * j] = ur - vr;     y[2 * j + 1] = ui - vi;
            }
        }
    }

    if (invert) {
        const double s = 1.0 / static_cast<double>(n);
        for (std::size_t i = 0; i < 2 * n; ++i) d[i] *= s;
    }
}

// ----- Weighted-digit machinery for Mersenne squaring ------------------------

// log2(n) + 2*(bits per digit) must stay below this so the largest convolution
// coefficient (including the weights) stays well inside a double's 53 bits.
constexpr long double kMaxCoeffBits = 46.0L;

unsigned bitLength(std::uint64_t v) {      // v != 0
    return 64u - static_cast<unsigned>(__builtin_clzll(v));
}

// Returns the (cached, per-thread) plan for exponent p. The returned reference
// stays valid until the next getPlan() call with a different p/boost.
const MersennePlan& getPlan(std::uint64_t p, std::size_t boost) {
    thread_local MersennePlan plan;
    if (plan.n != 0 && plan.p == p && plan.boost == boost) return plan;

    std::size_t n = 1;
    for (;;) {
        const std::uint64_t maxc = (p + n - 1) / n;        // ceil(p/n)
        const long double lb = std::log2(static_cast<long double>(n))
                             + 2.0L * static_cast<long double>(maxc);
        if (n >= 2 && lb < kMaxCoeffBits) break;
        n <<= 1;
    }
    n <<= boost;

    plan.p = p;
    plan.boost = boost;
    plan.n = n;
    plan.lo.assign(n + 1, 0);
    plan.width.assign(n, 0);
    plan.w.assign(n, 1.0);
    plan.winv.assign(n, 1.0);
    for (std::size_t i = 0; i <= n; ++i)
        plan.lo[i] = static_cast<std::size_t>(((unsigned __int128)i * p + n - 1) / n);  // ceil
    for (std::size_t i = 0; i < n; ++i) {
        plan.width[i] = static_cast<unsigned>(plan.lo[i + 1] - plan.lo[i]);
        const unsigned __int128 rem = ((unsigned __int128)i * p) % n;
        const unsigned __int128 k = (n - rem) % n;         // lo[i]*n - i*p, in [0, n-1]
        plan.w[i] = std::exp2(static_cast<double>(k) / static_cast<double>(n));
        plan.winv[i] = 1.0 / plan.w[i];
    }
    return plan;
}

// Carry-propagates signed digits into [0, 2^width[i]). The carry leaving the
// top digit has weight 2^p == 1 (mod 2^p - 1), so it wraps onto digit 0; that
// wrap IS the modular reduction. Also handles negative values (borrows).
void normalizeDigits(const MersennePlan& P, std::vector<long long>& g) {
    const std::size_t n = P.n;
    for (int pass = 0; pass < 64; ++pass) {
        long long carry = 0;
        for (std::size_t i = 0; i < n; ++i) {
            const long long v = g[i] + carry;
            const unsigned wd = P.width[i];
            g[i] = v & ((1LL << wd) - 1);
            carry = v >> wd;                               // arithmetic shift = floor
        }
        if (carry == 0) return;
        g[0] += carry;
    }
    throw std::logic_error("carry propagation did not converge");
}

// g := g^2 - subtract (mod 2^p - 1), in place, on normalized weighted digits.
// Returns the worst distance from an integer seen when rounding the inverse
// FFT output (a rounding-error gauge; > ~0.4 means the result can't be trusted).
double squareDigits(const MersennePlan& P, std::vector<long long>& g,
                    std::vector<std::complex<double>>& A, long long subtract) {
    const std::size_t n = P.n;
    A.resize(n);
    for (std::size_t i = 0; i < n; ++i)
        A[i] = std::complex<double>(static_cast<double>(g[i]) * P.w[i], 0.0);
    fft(A, false);
    for (std::size_t i = 0; i < n; ++i) {                  // A[i] = A[i]^2
        const double re = A[i].real(), im = A[i].imag();
        A[i] = std::complex<double>(re * re - im * im, 2.0 * re * im);
    }
    fft(A, true);

    double maxErr = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
        const double v = A[i].real() * P.winv[i];
        double r = std::nearbyint(v);
        if (!(std::fabs(r) < 9.0e18)) {                    // NaN / overflow: flag it, stay defined
            maxErr = HUGE_VAL;
            r = 0.0;
        }
        const double err = std::fabs(v - r);
        if (!(err <= maxErr)) maxErr = err;
        g[i] = static_cast<long long>(r);
    }
    g[0] -= subtract;
    normalizeDigits(P, g);
    return maxErr;
}

bool allDigitsMax(const MersennePlan& P, const std::vector<long long>& g) {
    for (std::size_t i = 0; i < P.n; ++i)
        if (g[i] != (1LL << P.width[i]) - 1) return false;
    return true;
}

// ----- Checkpoints (portable: stores the value's bits, not the digits) -------

const char kCkptMagic[8] = {'L', 'L', 'C', 'K', '0', '0', '0', '1'};

void saveCheckpoint(const std::string& path, std::uint64_t p, std::uint64_t iter,
                    const ByteArray& s) {
    const std::string tmp = path + ".tmp";
    {
        std::ofstream f(tmp, std::ios::binary | std::ios::trunc);
        if (!f) return;
        const std::uint64_t nbits = s.size();
        f.write(kCkptMagic, 8);
        f.write(reinterpret_cast<const char*>(&p), 8);
        f.write(reinterpret_cast<const char*>(&iter), 8);
        f.write(reinterpret_cast<const char*>(&nbits), 8);
        std::string bytes((nbits + 7) / 8, '\0');
        for (std::size_t i = 0; i < nbits; ++i)
            if (s.getBit(i)) bytes[i / 8] |= static_cast<char>(1u << (7 - i % 8));
        f.write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
        f.flush();
        if (!f) return;                                    // keep the previous checkpoint
    }
    if (std::rename(tmp.c_str(), path.c_str()) != 0) {     // Windows can't rename over a file
        std::remove(path.c_str());
        std::rename(tmp.c_str(), path.c_str());
    }
}

bool loadCheckpoint(const std::string& path, std::uint64_t p, std::uint64_t total,
                    std::uint64_t& iter, ByteArray& s) {
    std::ifstream f(path, std::ios::binary);
    if (!f) return false;
    char magic[8];
    std::uint64_t fp = 0, fi = 0, nbits = 0;
    f.read(magic, 8);
    f.read(reinterpret_cast<char*>(&fp), 8);
    f.read(reinterpret_cast<char*>(&fi), 8);
    f.read(reinterpret_cast<char*>(&nbits), 8);
    if (!f || std::memcmp(magic, kCkptMagic, 8) != 0) return false;
    if (fp != p || fi > total || nbits == 0 || nbits > p) return false;
    std::string bytes((nbits + 7) / 8, '\0');
    f.read(&bytes[0], static_cast<std::streamsize>(bytes.size()));
    if (!f) return false;
    ByteArray t;
    for (std::size_t i = 0; i < nbits; ++i)
        t.append((static_cast<unsigned char>(bytes[i / 8]) >> (7 - i % 8)) & 1u);
    s = t;
    iter = fi;
    return true;
}

}  // namespace

// ---------------------------------------------------------------------------
// ByteArray
// ---------------------------------------------------------------------------

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

std::uint64_t ByteArray::readBits(std::size_t start, std::size_t len) const {
    std::uint64_t v = 0;
    std::size_t i = start;
    const std::size_t end = start + len;
    while (i < end) {                       // take up to a byte boundary at a time
        const std::size_t off = i % 8;
        const std::size_t take = std::min<std::size_t>(8 - off, end - i);
        const unsigned b = (data_[i / 8] >> (8 - off - take)) & ((1u << take) - 1u);
        v = (v << take) | b;
        i += take;
    }
    return v;
}

void ByteArray::writeBits(std::size_t start, std::size_t len, std::uint64_t v) {
    std::size_t end = start + len;          // fill from the least significant end
    while (len > 0) {
        const std::size_t last = end - 1;
        const std::size_t off = last % 8;
        const std::size_t take = std::min<std::size_t>(len, off + 1);
        data_[last / 8] |= static_cast<std::uint8_t>((v & ((1u << take) - 1u)) << (7 - off));
        v >>= take;
        len -= take;
        end -= take;
    }
}

int ByteArray::compare(const ByteArray& other) const {
    std::size_t lenA = 0, aTop = 0, lenB = 0, bTop = 0;
    for (std::size_t i = 0; i < size_; ++i)
        if (bit(i)) { lenA = size_ - i; aTop = i; break; }
    for (std::size_t i = 0; i < other.size_; ++i)
        if (other.bit(i)) { lenB = other.size_ - i; bTop = i; break; }

    if (lenA != lenB) return lenA > lenB ? 1 : -1;
    for (std::size_t k = 0; k < lenA; ++k) {
        const bool ba = bit(aTop + k), bb = other.bit(bTop + k);
        if (ba != bb) return ba ? 1 : -1;
    }
    return 0;
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
    if (compare(other) < 0) throw std::invalid_argument("minus: this < other");
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

// Splits this (< 2^p) into the plan's weighted digits. A digit's bits live at
// little-endian positions [lo, lo+width), i.e. MSB-first positions
// [size_-lo-width, size_-lo), so each digit is one short multi-byte read.
void ByteArray::toDigits(const MersennePlan& P, std::vector<long long>& g) const {
    g.assign(P.n, 0);
    for (std::size_t i = 0; i < P.n; ++i) {
        const std::size_t lo = P.lo[i];
        if (lo >= size_) break;                            // everything above is zero
        const std::size_t avail = std::min<std::size_t>(P.width[i], size_ - lo);
        g[i] = static_cast<long long>(readBits(size_ - lo - avail, avail));
    }
}

// Inverse of toDigits (digits must be normalized). Result has no leading zeros.
ByteArray ByteArray::fromDigits(const MersennePlan& P, const std::vector<long long>& g) {
    ByteArray r;
    std::size_t highest = 0;
    bool any = false;
    for (std::size_t i = P.n; i-- > 0;) {
        if (g[i]) {
            highest = P.lo[i] + bitLength(static_cast<std::uint64_t>(g[i])) - 1;
            any = true;
            break;
        }
    }
    if (!any) { r.setZero(); return r; }

    r.size_ = highest + 1;
    r.data_.assign((r.size_ + 7) / 8, 0);
    for (std::size_t i = 0; i < P.n; ++i) {
        if (!g[i]) continue;
        const std::size_t len = bitLength(static_cast<std::uint64_t>(g[i]));
        r.writeBits(r.size_ - P.lo[i] - len, len, static_cast<std::uint64_t>(g[i]));
    }
    return r;
}

ByteArray ByteArray::squareModMersenne(std::size_t p) const {
    if (size_ > p) return modMersenne(p).squareModMersenne(p);

    const MersennePlan& P = getPlan(p, 0);
    std::vector<long long> g;
    toDigits(P, g);

    std::vector<std::complex<double>> A;
    const double err = squareDigits(P, g, A, 0);
    if (!(err <= 0.4))
        throw RoundoffError("squareModMersenne: FFT roundoff " + std::to_string(err) +
                            " too large for p = " + std::to_string(p));

    if (allDigitsMax(P, g)) {                              // value == 2^p - 1 == 0 (mod M)
        ByteArray z;
        z.setZero();
        return z;
    }
    return fromDigits(P, g);
}

LLResult ByteArray::lucasLehmer(std::uint64_t p, const LLOptions& opt) {
    LLResult res;
    if (p < 2) return res;
    if (p == 2) { res.isPrime = true; return res; }       // 3 is prime (the loop needs p odd)

    const MersennePlan& P = getPlan(p, opt.fftBoost);
    const std::uint64_t total = p - 2;
    const bool ckpt = !opt.checkpointFile.empty();

    std::vector<long long> g;
    std::uint64_t iter = 0;
    ByteArray saved;
    if (ckpt && loadCheckpoint(opt.checkpointFile, p, total, iter, saved)) {
        saved.toDigits(P, g);
        res.resumedFrom = iter;
    } else {
        iter = 0;
        g.assign(P.n, 0);
        g[0] = 4;                                          // s0 = 4
        normalizeDigits(P, g);
    }

    std::vector<std::complex<double>> A(P.n);
    while (iter < total) {
        const double e = squareDigits(P, g, A, 2);         // s = s^2 - 2 (mod M)
        if (!(e <= opt.maxAllowedRoundoff))
            throw RoundoffError("Lucas-Lehmer p = " + std::to_string(p) + ": FFT roundoff " +
                                std::to_string(e) + " at iteration " +
                                std::to_string(iter + 1) + " (try a larger fftBoost)");
        if (e > res.maxRoundoff) res.maxRoundoff = e;
        ++iter;

        if (ckpt && opt.checkpointEvery && iter % opt.checkpointEvery == 0 && iter < total)
            saveCheckpoint(opt.checkpointFile, p, iter, fromDigits(P, g));
        if (opt.onProgress && opt.progressEvery && iter % opt.progressEvery == 0)
            opt.onProgress(iter, total, res.maxRoundoff);
    }

    // M_p is prime iff s_{p-2} == 0 (mod M): all digits zero, or all ones (= M itself).
    bool allZero = true;
    for (std::size_t i = 0; i < P.n; ++i)
        if (g[i] != 0) { allZero = false; break; }
    res.isPrime = allZero || allDigitsMax(P, g);

    if (ckpt) std::remove(opt.checkpointFile.c_str());
    return res;
}

bool ByteArray::greaterEqual(const ByteArray& other) const { return compare(other) >= 0; }

bool ByteArray::isLargerThan(const ByteArray& other) const { return compare(other) > 0; }

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