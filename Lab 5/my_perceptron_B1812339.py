import pandas as pd
import numpy as np

def my_perceptron(x, y, eta, loop_count):
    x.insert(0, "X0", np.ones(len(x)))

    m, n = x.shape
    print("m =", m, "n =", n)

    w = np.round(np.random.rand(n), 3)
    print(" w=", w)

    for cnt in range(loop_count):
        print("lan lap ____", cnt+1)
        for i in range(m):
            gx = sum(x.iloc[i]*w)
            print("gx =", gx)
            output = int(gx > 0)
            w = w+eta*(y.iloc[i]-output)*x.iloc[i].to_numpy()
            print(" w=", w)
    return np.round(w, 3)

if __name__ == "__main__":
    df = pd.read_csv("data_per.csv", index_col=1)
    x = df.drop('Y', axis=1)
    y = df['Y']
    my_perceptron(x, y, 0.15, 2)