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


def average_growth_of(L: np.array, N, eradication=0) -> float:
    values = get_ps_n(L=L, N=N, eradication=eradication)

    differences = []
    for i in range(1, N):
        differences.append(values[i] / values[i - 1])

    return np.mean(differences)


def L(
    *args,
    culling_rate: float = 0,
    birth_control: float = 0,
    birth_rates=birth_rates,
    survival_rates=survival_rates,
) -> np.ndarray:
    culling_rate = float(culling_rate)
    birth_control = float(birth_control)
    # take into account birth_control, * (1 - birth_control)
    b = birth_rates * (1 - birth_control)
    # assert birth_rates[1] == 2.37

    assert culling_rate >= 0 and culling_rate <= 1
    assert birth_control >= 0 and birth_control <= 1

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


def get_p_n(L, *args, N=20, eradication=0) -> float:
    assert eradication >= 0 and eradication <= 1
    L_n = linalg.matrix_power(L, N)
    P_n = np.matmul(L_n, initial)
    e = (1 - eradication) ** N
    return np.sum(P_n) * e


def get_ps_n(L, N=20, eradication: float = 0.0):
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
    *args,
    target_population: float = 50,
    N=20,
    survival_rates=survival_rates,
    birth_rates=birth_rates,
    accuracy: float = 0.0001,
) -> (str, float):
    # iterate through each, get_p_n, stop if population is less than target_population
    # return the cull rate that got closest to target_population
    culls = np.arange(0, 1, accuracy)
    # print(f"{culls=}")
    for c in culls:
        L_culled = L(
            culling_rate=c, survival_rates=survival_rates, birth_rates=birth_rates
        )
        P_n = get_p_n(L_culled, N)
        if P_n < target_population:
            print(f"Optimal cull results: {P_n=}, {c=}")
            result = f"The optimal cull to reach a target population of {target_population} thousand is {c} ± {accuracy}. This reached a population of {P_n} thousand after {N} iterations = after {N * 5} years."
            return (result, c)

    return None


def find_stable_cull(
    *args,
    N=500,
    survival_rates=survival_rates,
    birth_rates=birth_rates,
    accuracy=0.005,
) -> (str, float):
    # iterate through each, get_p_n, stop if population is less than target_population
    # return the cull rate that got closest to target_population
    culls = np.arange(0, 1, accuracy)
    # print(f"{culls=}")

    closest_so_far = 1000000
    """Distance from growth rate of 1"""
    closest_so_far_c = 0
    """Cull rate that was tried last"""
    for c in culls:
        L_culled = L(c, survival_rates=survival_rates, birth_rates=birth_rates)
        avg_growth = average_growth_of(L=L_culled, N=N)
        distance_from_stable = abs(avg_growth - 1)
        if distance_from_stable < closest_so_far:
            closest_so_far = distance_from_stable
            closest_so_far_c = c
        else:
            avg = average_growth_of(L=L(), N=N)
            print(f"Stable cull results: {closest_so_far_c=} {closest_so_far=} {avg=}")
            previous_p_n = get_p_n(
                L=L(
                    culling_rate=closest_so_far_c,
                    survival_rates=survival_rates,
                    birth_rates=birth_rates,
                ),
                N=N,
            )
            result = f"The cull rate that is most stable is {closest_so_far_c} ± {accuracy}. This cull rate had an average growth rate of {avg} and reached {previous_p_n} after {N} iterations = after {N * 5} years."
            return (result, closest_so_far_c)

    return None


def find_optimal_eradication(
    *args,
    accuracy=0.0001,
    target_population: float = 50,
    N=20,
    survival_rates=survival_rates,
    birth_rates=birth_rates,
) -> (str, float):
    # iterate through each, get_p_n, stop if population is less than target_population
    # return the cull rate that got closest to target_population
    eradication_rates = np.arange(0, 1, accuracy)
    # print(f"{culls=}")
    for e in eradication_rates:
        L_modified = L(survival_rates=survival_rates, birth_rates=birth_rates)
        P_n = get_p_n(L=L_modified, N=N, eradication=e)
        if P_n < target_population:
            print(f"Optimal eradication results: {P_n=}, {e=}")
            result = f"The optimal eradication rate to reach a target population of {target_population} thousand is {e} ± {accuracy}. This reached a population of {P_n} thousand after {N} iterations = after {N * 5} years."
            return (result, e)

    return None


def find_stable_eradication(
    *args,
    accuracy=0.005,
    N=500,
    survival_rates=survival_rates,
    birth_rates=birth_rates,
) -> (str, float):
    # iterate through each, get_p_n, stop if population is less than target_population
    # return the cull rate that got closest to target_population
    eradication_rates = np.arange(0, 1, accuracy)
    # print(f"{culls=}")
    closest_so_far = 1000000
    closest_so_far_e = 0
    for e in eradication_rates:
        L_modified = L(survival_rates=survival_rates, birth_rates=birth_rates)
        avg_growth = average_growth_of(L=L_modified, N=N, eradication=e)
        distance_from_stable = abs(avg_growth - 1)
        if distance_from_stable < closest_so_far:
            closest_so_far = distance_from_stable
            closest_so_far_e = e
        else:
            avg = average_growth_of(L=L(), N=N, eradication=closest_so_far_e)
            print(
                f"Stable eradication results: {closest_so_far_e=} {closest_so_far=} {avg=}"
            )
            previous_p_n = get_p_n(
                L=L(
                    survival_rates=survival_rates,
                    birth_rates=birth_rates,
                ),
                N=N,
                eradication=closest_so_far_e,
            )
            result = f"The eradication rate that is most stable is {closest_so_far_e} ± {accuracy}. This eradication rate had an average growth rate of {avg} and reached {previous_p_n} after {N} iterations = after {N * 5} years."
            return (result, closest_so_far_e)

    return None


def find_optimal_birth_control(
    *args,
    accuracy=0.0001,
    target_population: float = 50,
    N=20,
    survival_rates=survival_rates,
    birth_rates=birth_rates,
) -> (str, float):
    # iterate through each, get_p_n, stop if population is less than target_population
    # return the birth control rate that got closest to target_population
    birth_controls = np.arange(0, 1, accuracy)
    # print(f"{birth_controls=}")
    for b in birth_controls:
        L_modified = L(
            birth_control=b, survival_rates=survival_rates, birth_rates=birth_rates
        )
        P_n = get_p_n(L_modified, N)
        if P_n <= target_population:
            print(f"Optimal Birth Control result: {P_n=}, {b=}")
            result = f"The optimal birth control rate to reach a target population of {target_population} thousand is {b} ± {accuracy}. This reached a population of {P_n} thousand after {N} iterations = after {N * 5} years."
            return (result, b)

    return None


def find_stable_birth_control(
    *args,
    accuracy=0.005,
    N=500,
    survival_rates=survival_rates,
    birth_controls=birth_rates,
) -> (str, float):
    # iterate through each, get_p_n, stop if population is less than target_population
    # return the cull rate that got closest to target_population
    birth_controls = np.arange(0, 1, accuracy)
    # print(f"{culls=}")
    closest_so_far = 1000000.0
    closest_so_far_b = 0.0

    for b in birth_controls:
        L_modified = L(
            birth_control=b, survival_rates=survival_rates, birth_rates=birth_rates
        )
        avg_growth = average_growth_of(L=L_modified, N=N)

        distance_from_stable = abs(avg_growth - 1)

        if distance_from_stable < closest_so_far:
            closest_so_far = distance_from_stable
            closest_so_far_b = b
        else:
            previous_L = L(
                birth_control=closest_so_far_b,
                birth_rates=birth_rates,
                survival_rates=survival_rates,
            )
            avg = average_growth_of(
                L=previous_L,
                N=N,
            )
            print(
                f"Stable birth control results: {closest_so_far_b=} {closest_so_far=} {distance_from_stable=} {avg=}"
            )
            previous_p_n = get_p_n(
                L=previous_L,
                N=N,
            )
            result = f"The birth control rate that is most stable is {closest_so_far_b} ± {accuracy}. This birth control rate had an average growth rate of {avg} and reached {previous_p_n} after {N} iterations = after {N * 5} years."
            return (result, closest_so_far_b)

    return None


def write_cull_rates_to_csv(
    *args,
    median,
    deviation=0.05,
    step=0.01,
    file_name: str = "cull_rates.csv",
    N=20,
    survival_rates=survival_rates,
    birth_rates=birth_rates,
):
    cull_rates = [
        median,
        *np.arange(median - deviation, median + deviation, step).tolist(),
    ]
    # iterate through each, get_p_n, stop if population is less than target_population
    # return the cull rate that got closest to target_population
    # print(f"{cull_rates=}")
    results = [["-", *map(lambda n: 5 * n, list(range(N + 1)))]]
    for c in cull_rates:
        L_culled = L(
            culling_rate=c, survival_rates=survival_rates, birth_rates=birth_rates
        )
        Ps_n = get_ps_n(L_culled, N)
        results.append([c, *Ps_n])

    columns = ["Cull Rate", *map(lambda n: 5 * n, list(range(N + 1)))]
    df = pd.DataFrame(results, columns=columns)
    df.to_csv(file_name, index=False)


def write_birth_controls_to_csv(
    *args,
    median,
    deviation=0.05,
    step=0.01,
    file_name: str = "birth_controls.csv",
    N=20,
    survival_rates=survival_rates,
    birth_rates=birth_rates,
):
    birth_controls = [
        median,
        *np.arange(median - deviation, median + deviation, step).tolist(),
    ]
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


def write_eradication_rates_to_csv(
    *args,
    median,
    deviation=0.05,
    step=0.01,
    file_name: str = "eradication_rates.csv",
    N=20,
    survival_rates=survival_rates,
    birth_rates=birth_rates,
):
    eradication_rates = [
        median,
        *np.arange(median - deviation, median + deviation, step).tolist(),
    ]
    # iterate through each, get_p_n, stop if population is less than target_population
    # return the birth control rate that got closest to target_population
    # print(f"{birth_controls=}")
    results = [["-", *map(lambda n: 5 * n, list(range(N + 1)))]]
    for e in eradication_rates:
        L_modified = L(survival_rates=survival_rates, birth_rates=birth_rates)
        Ps_n = get_ps_n(L=L_modified, N=N, eradication=e)
        results.append([e, *Ps_n])

    columns = ["Eradication Rate", *map(lambda n: 5 * n, list(range(N + 1)))]
    df = pd.DataFrame(results, columns=columns)
    df.to_csv(file_name, index=False)


optimal_cull = find_optimal_cull()
print(f"{optimal_cull=}")
write_cull_rates_to_csv(
    median=optimal_cull[1],
    file_name="cull_rates_optimum.csv",
)

optimal_eradication = find_optimal_eradication()
print(f"{optimal_eradication=}")
write_eradication_rates_to_csv(
    median=optimal_eradication[1],
    file_name="eradication_rates_optimum.csv",
)

stable_eradication = find_stable_eradication()
print(f"{stable_eradication=}")
write_eradication_rates_to_csv(
    median=stable_eradication[1],
    file_name="eradication_rates_stable.csv",
)

optimal_birth_control = find_optimal_birth_control()
print(f"{optimal_birth_control=}")
write_birth_controls_to_csv(
    median=optimal_birth_control[1],
    file_name="birth_controls_optimum.csv",
)

stable_birth_control = find_stable_birth_control()
print(f"{stable_birth_control=}")
write_birth_controls_to_csv(
    median=stable_birth_control[1],
    file_name="birth_controls_stable.csv",
)
