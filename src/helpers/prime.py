import math
from typing import Optional


FIRST_PRIMES = [
    2,
    3,
    5,
    7,
    11,
    13,
    17,
    19,
    23,
    29,
    31,
    37,
    41,
    43,
    47,
    53,
    59,
    61,
    67,
    71,
    73,
    79,
    83,
    89,
    97,
    101,
    103,
    107,
    109,
    113,
    127,
    131,
    137,
    139,
    149,
    151,
    157,
    163,
    167,
    173,
    179,
    181,
    191,
    193,
    197,
    199,
    211,
    223,
    227,
    229,
    233,
    239,
    241,
    251,
    257,
    263,
    269,
    271,
    277,
    281,
    283,
    293,
    307,
    311,
    313,
    317,
    331,
    337,
    347,
    349,
]


def prime_factors(n: int, max_prime: Optional[int] = None) -> set[int]:
    if max_prime is None:
        max_prime = int(math.sqrt(n)) + 1
    prime_check_range = range(2, max_prime)

    prime_factor_set = set()
    for i in prime_check_range:
        while n % i == 0:
            n = n // i
            prime_factor_set.add(i)
    if n != 1:
        prime_factor_set.add(n)
    return prime_factor_set
