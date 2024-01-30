import numpy.linalg as linalg
from numpy.linalg import eig
import numpy as np
import pandas as pd

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


def average_growth_of(L: np.array, N) -> float:
    values = []
    for i in range(N):
        L_n = linalg.matrix_power(L, i)
        P_n = np.matmul(L_n, initial)
        sum = np.sum(P_n)
        values.append(sum)

    differences = []
    for i in range(1, N):
        differences.append(values[i] / values[i - 1])

    return np.mean(differences)


def L(
    culling_rate: float = 0,
    birth_control: float = 0,
    birth_rates=birth_rates,
    survival_rates=survival_rates,
) -> np.ndarray:
    culling_rate = float(culling_rate)
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


def get_p_n(L, N=20, eradication=0) -> float:
    assert eradication >= 0 and eradication <= 1
    L_n = linalg.matrix_power(L, N)
    P_n = np.matmul(L_n, initial)
    e = (1 - eradication) ** N
    return np.sum(P_n) * e


def get_ps_n(L, N=20, eradication=0):
    """0 - 20, len() == 21"""
    assert eradication >= 0 and eradication <= 1
    ret = []
    for i in range(N + 1):
        L_n = linalg.matrix_power(L, i)
        P_n = np.matmul(L_n, initial)
        e = (1 - eradication) ** i
        ret.append(np.sum(P_n) * e)
    return ret


def find_optimal_cull(
    target_population: float = 50,
    N=20,
    survival_rates=survival_rates,
    birth_rates=birth_rates,
):
    # iterate through each, get_p_n, stop if population is less than target_population
    # return the cull rate that got closest to target_population
    culls = np.arange(0.55, 0.65, 0.0001)
    # print(f"{culls=}")
    for c in culls:
        L_culled = L(c, survival_rates=survival_rates, birth_rates=birth_rates)
        P_n = get_p_n(L_culled, N)
        if P_n < target_population:
            print(f"{P_n=}, {c=}")
            return c

    return None


print(f"{find_optimal_cull()=}")


def find_optimal_birth_control(
    target_population: float = 50,
    N=20,
    survival_rates=survival_rates,
    birth_rates=birth_rates,
):
    # iterate through each, get_p_n, stop if population is less than target_population
    # return the birth control rate that got closest to target_population
    birth_controls = np.arange(0.65, 0.75, 0.0001)
    # print(f"{birth_controls=}")
    for b in birth_controls:
        L_modified = L(
            birth_control=b, survival_rates=survival_rates, birth_rates=birth_rates
        )
        P_n = get_p_n(L_modified, N)
        if P_n <= target_population:
            print(f"{P_n=}, {b=}")
            return b

    return None


print(f"{find_optimal_birth_control()=}")


def write_birth_controls_to_csv(
    birth_controls=np.arange(0.65, 0.75, 0.01),
    file_name: str = "birth_controls.csv",
    N=20,
    survival_rates=survival_rates,
    birth_rates=birth_rates,
):
    # iterate through each, get_p_n, stop if population is less than target_population
    # return the birth control rate that got closest to target_population
    # print(f"{birth_controls=}")
    results = [["-", *map(lambda n: 5 * n, list(range(N + 1)))]]
    for b in birth_controls:
        L_modified = L(
            birth_control=b, survival_rates=survival_rates, birth_rates=birth_rates
        )
        Ps_n = get_ps_n(L_modified, N)
        results.append([b, *Ps_n])

    columns = ["Birth Control Rate", *map(lambda n: 5 * n, list(range(N + 1)))]
    df = pd.DataFrame(results, columns=columns)
    df.to_csv(file_name, index=False)


write_birth_controls_to_csv()
