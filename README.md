# Mersenne Primes via Array Math

An experiment in finding [Mersenne primes](https://en.wikipedia.org/wiki/Mersenne_prime) —
primes of the form `2^p - 1` — without ever storing the huge number itself.

Mersenne primes have two useful properties that this project leans on:

1. `2^p - 1`, viewed as a bit array, is just `p` ones (`111...111`).
2. For `2^p - 1` to be prime, `p` must itself be prime.

The numbers grow far beyond what a machine can store as native integers, so this
project keeps everything as **bit arrays** and does the math directly on the bits
(compare, subtract, square-root, modulo). No big integer is ever materialized.

## How it works

`mp.cpp` defines a `ByteArray` class providing array-math primitives
(`approxSqrt`, `remainder`/`isDivisibleBy`, `addTwo`, `isLargerThan`, ...).
`main` searches for Mersenne primes like this:

1. Outer loop over odd prime `p` (starting at `p = 3` and up).
2. Build `m` = bit array of `p` ones (= `2^p - 1`).
3. Test whether `m` is prime:
   - compute `sqrt(m)` as a bit array,
   - trial-divide `m` by every odd `f` from `3` up to `sqrt(m)`,
   - if any `f` divides `m`, `m` is composite → back to the outer loop,
   - if none divides `m`, `2^p - 1` is prime.

## Build

```sh
g++ -std=c++11 mp.cpp -o mp
```

## Run

```sh
./mp <start_p> <end_p>
```

For each odd prime `p` in `[start_p, end_p]`, the program builds `2^p - 1` and
prints `2^p-1 is prime!` when it finds a Mersenne prime, or `p = <p> : composite`
otherwise.

Examples:

```sh
./mp 3 30     # finds 2^3-1, 2^5-1, 2^7-1, 2^13-1, 2^17-1, 2^19-1, 2^31-1*
```

> *Note: the trial-division search is exponential in `p`, so it becomes
> impractically slow beyond roughly `p = 61`. Keep ranges small.

## Known Mersenne prime exponents

`2, 3, 5, 7, 13, 17, 19, 31, 61, 89, 107, 127, 521, 607, 1279, ...`

## Author

zpollock@gmail.com
