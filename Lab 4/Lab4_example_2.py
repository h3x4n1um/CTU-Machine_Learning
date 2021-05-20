from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

if __name__ == "__main__":
    dt = pd.read_csv("Housing_2019.csv", index_col=0)
    dt.iloc[2:4, ]
    X = dt.iloc[:, [1, 2, 3, 4, 10]]
    y = dt.price
    plt.scatter(dt.lotsize, dt.price)
    plt.show()

    lrm = LinearRegression()
    lrm.fit(X[0:520], y[0:520])

    print(lrm.intercept_)
    print(lrm.coef_)

    y_test = y[-20:]
    X_test = X[-20:]
    y_pred = lrm.predict(X_test)

    print(y_pred)
    print(y_test)

    err = mean_squared_error(y_test, y_pred)
    print(err)
    rmse_err = np.sqrt(err)
    print(round(rmse_err, 3))