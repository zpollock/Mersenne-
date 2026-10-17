# Mersenne Primes via Array Math

An experiment in finding [Mersenne primes](https://en.wikipedia.org/wiki/Mersenne_prime).

Mersenne primes rely on two neat properties:

1. `2^p - 1`, viewed as a bit array, is just `p` ones (`111...111`).
2. For `2^p - 1` to be prime, `p` must itself be prime.

The numbers grow far beyond what a machine can store as native integers, so this
project keeps everything as **bit arrays** and does the math directly on the bits
(compare, subtract, multiply, add, modulo). No big integer is ever materialized.

The library stores its values bit-packed (8 bits per byte), so a `p`-bit number
takes only about `p / 8` bytes.

## How it works

The array math lives in a small `ByteArray` library (`bytearray.h` / `bytearray.cpp`),
and `main` (in `mp.cpp`) runs the search.

Each Mersenne prime `2^p - 1` is tested with the [Lucas-Lehmer test](https://en.wikipedia.org/wiki/Lucas%E2%80%93Lehmer_primality_test):

1. Outer loop over odd primes `p`, starting at `p = 3`.
2. Start with `s = 4`.
3. Repeat `p - 2` times: replace `s` with `(s^2 - 2) mod (2^p - 1)`.
4. If `s` ends up at 0, then `2^p - 1` is prime.

Since `2^p` is just 1 more than `2^p - 1`,
a square that is up to `2p` bits long can be reduced by folding its top `p` bits
into its bottom `p` bits. Nothing ever grows beyond `2p` bits, so no big numbers
appear anywhere.

This is quadratic in `p`, which is dramatically faster than trial division. Earlier
versions tried dividing `2^p - 1` by every odd number up to its square root, which
grows exponentially and effectively stalls past `p = 61`.

## Build

```sh
g++ -std=c++11 mp.cpp bytearray.cpp -o mp
```

## Run

```sh
./mp <start_p> <end_p>
```

For each odd prime `p` in `[start_p, end_p]`, the program runs the Lucas-Lehmer
test and prints `2^p-1 is prime!` when it finds a Mersenne prime, or
`p = <p> : composite` otherwise.

Examples:

```sh
./mp 3 60        # finds 2^3-1 through 2^61-1*
./mp 61 127      # finds 2^61-1, 2^89-1, 2^107-1, 2^127-1
```

> The search is now fast enough to cover the small Mersenne primes quickly. As
> `p` grows it still slows down, but far less steeply than trial division did.

## Known Mersenne prime exponents

`2, 3, 5, 7, 13, 17, 19, 31, 61, 89, 107, 127, 521, 607, 1279, ...`

## Author

zpollock@gmail.com
