from multiprocessing import Pool
from time import time
from logging import INFO, log, info, basicConfig

start_time = time()

basicConfig(level=INFO)


def f(i):
    print("hi")
    while start_time + i * 5 > time():
        pass
    return f"{start_time + 5} {time()}"


with Pool(5) as p:
    print(p.map(f, [1, 2, 3, 4, 5]))
