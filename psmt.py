import numpy.linalg as linalg
from numpy.linalg import eig
import numpy as np
import pandas as pd
import sys

dataset_num = int([*sys.argv, 14][1])
print(f"Using data set number: {dataset_num=}")

initial_data = np.array(
    [
        [170000, 90000, 75000, 60000, 49000, 30000, 21000, 5000],
        [165000, 95000, 77000, 58000, 48950, 30050, 21000, 5000],
        [160000, 100000, 79000, 56000, 48900, 30100, 21000, 5000],
        [155000, 105000, 81000, 54000, 48850, 30150, 21000, 5000],
        [150000, 110000, 83000, 52000, 48800, 30200, 21000, 5000],
        [145000, 115000, 85000, 50000, 48750, 30250, 21000, 5000],
        [140000, 120000, 87000, 48000, 48700, 30300, 21000, 5000],
        [135000, 125000, 89000, 46000, 48650, 30350, 21000, 5000],
        [130000, 130000, 91000, 44000, 48600, 30400, 21000, 5000],
        [125000, 135000, 93000, 42000, 48550, 30450, 21000, 5000],
        [120000, 140000, 95000, 40000, 48500, 30500, 21000, 5000],
        [115000, 145000, 97000, 38000, 48450, 30550, 21000, 5000],
        [110000, 150000, 99000, 36000, 48400, 30600, 21000, 5000],
        [105000, 155000, 101000, 34000, 48350, 30650, 21000, 5000],
    ]
)
initial = initial_data[dataset_num - 1] / 1000

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
    culls = np.arange(0, 1, accuracy)
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
    culls = np.arange(0, 1, accuracy)
    closest_so_far = 1000000
    """Distance from growth rate of 1"""
    closest_so_far_c = 0
    """Cull rate that was tried last"""
    for c in culls:
        L_culled = L(
            culling_rate=c, survival_rates=survival_rates, birth_rates=birth_rates
        )
        avg_growth = average_growth_of(L=L_culled, N=N)
        distance_from_stable = abs(avg_growth - 1)
        if distance_from_stable < closest_so_far:
            closest_so_far = distance_from_stable
            closest_so_far_c = c
        else:
            previous_l = L(
                culling_rate=closest_so_far_c,
                survival_rates=survival_rates,
                birth_rates=birth_rates,
            )
            avg = average_growth_of(L=previous_l, N=N)
            print(f"Stable cull results: {closest_so_far_c=} {closest_so_far=} {avg=}")
            previous_p_n = get_p_n(
                L=previous_l,
                N=N,
            )
            result = f"The cull rate that is most stable is {closest_so_far_c} ± {accuracy}. This cull rate had an average growth rate of {avg} and reached {previous_p_n} thousand after {N} iterations = after {N * 5} years."
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
    eradication_rates = np.arange(0, 1, accuracy)
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
    eradication_rates = np.arange(0, 1, accuracy)
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
            result = f"The eradication rate that is most stable is {closest_so_far_e} ± {accuracy}. This eradication rate had an average growth rate of {avg} and reached {previous_p_n} thousand after {N} iterations = after {N * 5} years."
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
    birth_controls = np.arange(0, 1, accuracy)
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
    birth_controls = np.arange(0, 1, accuracy)
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
            result = f"The birth control rate that is most stable is {closest_so_far_b} ± {accuracy}. This birth control rate had an average growth rate of {avg} and reached {previous_p_n} thousand after {N} iterations = after {N * 5} years."
            return (result, closest_so_far_b)

    return None


def write_cull_rates_to_csv(
    *args,
    text="-",
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
    results = [[text, *map(lambda n: 5 * n, list(range(N + 1)))]]
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
    text="-",
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
    results = [[text, *map(lambda n: 5 * n, list(range(N + 1)))]]
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
    text="-",
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
    results = [[text, *map(lambda n: 5 * n, list(range(N + 1)))]]
    for e in eradication_rates:
        L_modified = L(survival_rates=survival_rates, birth_rates=birth_rates)
        Ps_n = get_ps_n(L=L_modified, N=N, eradication=e)
        results.append([e, *Ps_n])

    columns = ["Eradication Rate", *map(lambda n: 5 * n, list(range(N + 1)))]
    df = pd.DataFrame(results, columns=columns)
    df.to_csv(file_name, index=False)


def write_initial_population_vector_to_csv(
    initial,
):
    df = pd.DataFrame(initial)
    df.to_csv("initial_population_vector.csv", index=False)


# Figure 4
write_cull_rates_to_csv(
    median=0,
    deviation=0,
    file_name="cull_rates_no_cull.csv",
    text="No cull",
)

# Figure 5
optimal_cull = find_optimal_cull()
print(f"{optimal_cull=}")
write_cull_rates_to_csv(
    median=optimal_cull[1],
    file_name=f"cull_rates_optimum.csv",
    text=optimal_cull[0],
)
# Figure 6
stable_cull = find_stable_cull()
print(f"{stable_cull=}")
write_cull_rates_to_csv(
    text=stable_cull[0],
    median=stable_cull[1],
    file_name="cull_rates_stable.csv",
)

# Figure 7
optimal_eradication = find_optimal_eradication()
print(f"{optimal_eradication=}")
write_eradication_rates_to_csv(
    text=optimal_eradication[0],
    median=optimal_eradication[1],
    file_name="eradication_rates_optimum.csv",
)

# Figure 8
stable_eradication = find_stable_eradication()
print(f"{stable_eradication=}")
write_eradication_rates_to_csv(
    text=stable_eradication[0],
    median=stable_eradication[1],
    file_name="eradication_rates_stable.csv",
)

# Figure 9
optimal_birth_control = find_optimal_birth_control()
print(f"{optimal_birth_control=}")
write_birth_controls_to_csv(
    text=optimal_birth_control[0],
    median=optimal_birth_control[1],
    file_name="birth_controls_optimum.csv",
)

# Figure 10
stable_birth_control = find_stable_birth_control()
print(f"{stable_birth_control=}")
write_birth_controls_to_csv(
    text=stable_birth_control[0],
    median=stable_birth_control[1],
    file_name="birth_controls_stable.csv",
)

# Figure 11
write_cull_rates_to_csv(
    median=1, deviation=0, file_name="max_cull_rates.csv", text="Max cull"
)
write_eradication_rates_to_csv(
    median=1, deviation=0, file_name="max_eradication_rates.csv", text="Max eradication"
)
