import numpy.linalg as linalg
from numpy.linalg import eig
import numpy as np
import pandas as pd

num = int(cell(10, 1))
# print(f"{num=}")

culls = int(cell(11, 1))
cull_n = np.array(cells((10, 3), (10, 2 + culls)), dtype=np.single)
# print(f"{cull_n=}, {culls=}")

i = np.array(cells((1, 12), (1, 19)), dtype=np.single)
# L = np.array(cells((1, 4), (8, 11)), dtype=np.single)

birth_rates = np.array(cells((1, 4), (8, 4)), dtype=np.single)[0]
survival_rates = np.array(cells((1, 21), (7, 21)), dtype=np.single)[0]
# print(f"{birth_rates=}, {survival_rates=}")


def L(culling_rate: float) -> np.ndarray:
    culling_rate = float(culling_rate)
    b = birth_rates
    assert culling_rate >= 0 and culling_rate <= 1
    s = survival_rates * (1 - culling_rate)
    # 8x8 leslie matrix
    ret = np.array(
        [
            [b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7]],
            [s[0], 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, s[1], 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, s[2], 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, s[3], 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, s[4], 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, s[5], 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, s[6], 0.0],
        ]
    )

    w, v = eig(ret)
    eigen_value = max(w.tolist())
    print(f"For culling rate {culling_rate}, eigenvalue is {eigen_value}")

    return ret


ret = np.zeros((culls, num))
for c in range(culls):
    # taking into account culling
    cull_factor = cull_n[c][0]
    L_culled = L(cull_factor)
    # print(f"{L_culled=}")
    for n in range(num):
        L_n = linalg.matrix_power(L_culled, n)
        # print(f"{L_n=}, {i=}")
        population_vector = np.matmul(L_n, i)
        # print(f"Population vector: {population_vector}")
        Sum = np.sum(population_vector)
        # print(f"Sum: {Sum}")
        ret[c][n] = Sum

ret = ret.tolist()
# print(f"{ret=}")
ret
