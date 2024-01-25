import numpy as np
import numpy.linalg as linalg
import json
import pandas as pd

# 0.59, 0.49, 0.37, 0.24, 0.12, 0.05, 0.01, 0
L = np.array(
    [
        [
            0,
            2.37,
            3.15,
            2.23,
            1.3,
            0,
            0,
            0,
        ],
        [
            0.59,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
        ],
        [
            0,
            0.49,
            0,
            0,
            0,
            0,
            0,
            0,
        ],
        [
            0,
            0,
            0.37,
            0,
            0,
            0,
            0,
            0,
        ],
        [
            0,
            0,
            0,
            0.24,
            0,
            0,
            0,
            0,
        ],
        [
            0,
            0,
            0,
            0,
            0.12,
            0,
            0,
            0,
        ],
        [
            0,
            0,
            0,
            0,
            0,
            0.05,
            0,
            0,
        ],
        [
            0,
            0,
            0,
            0,
            0,
            0,
            0.01,
            0,
        ],
    ]
)
i = np.array([105000, 155000, 101000, 34000, 48350, 30650, 21000, 5000])
# L^n * i for values of n from 1 to 10
X_n = [np.dot(linalg.matrix_power(L, n), i) for n in range(1, 11)]

data = res

# Create a DataFrame
df = pd.DataFrame(data)

# Export the DataFrame to an Excel file
df.to_excel("output.xlsx", index=False)

print(res, linalg.matrix_power(L, 3))
