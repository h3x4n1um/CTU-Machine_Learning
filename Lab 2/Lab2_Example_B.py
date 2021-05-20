import numpy as np
import pandas as pd

if __name__ == "__main__":
    dt5 = pd.read_csv("iris_data.csv")
    print(dt5[1:5])
    print(len(dt5))
    print(dt5.petalLength[1:5])

    X = [
        [0, 0],
        [1, 0],
        [1, 1],
        [2, 1],
        [2, 1],
        [2, 0]
    ]
    Y = [0, 0, 0, 1, 1, 0]
    df = pd.DataFrame(
        np.array([[X[i][0], X[i][1], Y[i]] for i in range(len(X))]),
        columns=["X1", "X2", "nhan"]
    )
    print(df)