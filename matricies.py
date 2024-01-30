import numpy.linalg as linalg
from numpy.linalg import eig
import numpy as np
import pandas as pd

# num = int(cell(10, 1))
# # print(f"{num=}")

# culls = int(cell(11, 1))
# cull_n = np.array(cells((10, 3), (10, 2 + culls)), dtype=np.single)
# # print(f"{cull_n=}, {culls=}")

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

# growth_iters = int(cell(0, 25))

eigen_values = []


def average_growth_of(L: np.array, N) -> float:
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

    # print(f"Values: {values}")

    return np.mean(differences)


def L(
    culling_rate: float,
    birth_control: float = 0,
    birth_rates=birth_rates,
    survival_rates=survival_rates,
) -> np.ndarray:
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

    # w, v = eig(ret)
    # # find the maximum real part of the eigenvalues w
    # eigen_value = max(w.tolist(), key=lambda x: abs(x))
    # print(f"For culling rate {culling_rate}, eigenvalue is {eigen_value}")
    # if eigen_value.imag != 0:
    #     print(f"Warning: eigenvalue has imaginary part {eigen_value.imag}")
    # eigen_value = abs(eigen_value)

    # growth_rate = average_growth_of(ret, growth_iters)

    # eigen_values.append((culling_rate, eigen_value, growth_rate))

    return ret


def get_p_n(L, N=20) -> float:
    L_n = linalg.matrix_power(L, N)
    P_n = np.matmul(L_n, initial)
    return np.sum(P_n)


def find_optimal_cull(
    target_population: float = 50,
    N=20,
    survival_rates=survival_rates,
    birth_rates=birth_rates,
):
    # make culls an array of floats 0.7, 0.71, 0.72, ... 0.99
    # iterate through each, get_p_n, stop if population is less than target_population
    # return the cull rate that got closest to target_population
    culls = np.arange(0.5, 1, 0.01)
    print(f"{culls=}")
    for c in culls:
        L_culled = L(c)
        P_n = get_p_n(L_culled, N)
        if P_n < target_population:
            print(f"{P_n=}, {c=}")
            return c

    return None


print(f"{find_optimal_cull()=}")

# ret = np.zeros((culls, num))
# for c in range(culls):
#     # taking into account culling
#     cull_factor = cull_n[c][0]
#     L_culled = L(cull_factor)
#     # print(f"{L_culled=}")
#     for n in range(num):
#         debug = False
#         # if n <= 1:
#         #     debug = True

#         L_n = linalg.matrix_power(L_culled, n)
#         if debug:
#             print(f"{L_n=}, {initial=}")
#         population_vector = np.matmul(L_n, initial)
#         if debug:
#             print(f"Population vector: {population_vector}")
#         Sum = np.sum(population_vector)
#         # print(f"Sum: {Sum}")
#         ret[c][n] = Sum

# ret = ret.tolist()
# # print(f"{ret=}")

# print(f"{eigen_values=}")

# # ret
# eigen_values
