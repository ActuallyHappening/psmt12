import numpy as np
from numpy.linalg import eig
import numpy.linalg as linalg

initial = np.array(
    [
        [105],
        [155],
        [101],
        [34],
        [48.35],
        [30.65],
        [21],
        [5],
    ]
)

birth_rates = np.array([0, 2.37, 3.15, 2.23, 1.3, 0, 0, 0])

survival_rates = np.array([0.59, 0.49, 0.37, 0.24, 0.12, 0.05, 0.01])


def L(culling_rate: float, birth_control: float = 0) -> np.ndarray:
    culling_rate = float(culling_rate)

    # b = birth_rates
    # take into account birth_control, * (1 - birth_control)
    b = birth_rates * (1 - birth_control)

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

    return ret


def average_growth_of(L: np.array, N=20) -> float:
    # apply leslie matrix to initial and sum,
    # record the average growth rate
    # print(f"{L=}")
    # print(f"{initial=}")
    values = []
    for i in range(N):
        L_n = linalg.matrix_power(L, i)
        P_n = np.matmul(L_n, initial)
        sum = np.sum(P_n)
        values.append(sum)

    differences = []
    for i in range(1, N):
        differences.append(values[i] / values[i - 1])

    print(f"Values: {values}")

    return np.mean(differences)


def get_p_n(L, N=20):
    L_n = linalg.matrix_power(L, N)
    P_n = np.matmul(L_n, initial)
    return np.sum(P_n)


# # mean = average_growth_of(L(0))
# # print(f"{mean=}")

# num = int(cell(10, 1))
# # print(f"{num=}")

# culls = int(cell(11, 1))
# cull_n = np.array(cells((10, 3), (10, 2 + culls)), dtype=np.single)
# # print(f"{cull_n=}, {culls=}")
