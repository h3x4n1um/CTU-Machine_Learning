from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def LR1(X, Y, eta, loop_cnt, theta0, theta1):
    m = len(X)
    for k in range(loop_cnt):
        print("Lan lap: {}".format(k))
        for i in range(m):
            h_i = theta0 + theta1*X[i]

            theta0 = theta0 + eta*(Y[i]-h_i)*1
            theta1 = theta1 + eta*(Y[i]-h_i)*X[i]

            print("Phan tu thu {}\n\ty={}\n\th={}\n\ttheta0={}\n\ttheta1={}".format(i, Y[i], h_i, round(theta0, 3), round(theta1, 3)))
    return [round(theta0, 3), round(theta1, 3)]

def LR2(X, Y, eta, loop_cnt, theta0, theta1):
    m = len(X)
    for cnt in range(loop_cnt):
        h_i = theta0+theta1*X

        sigma0 = np.sum(Y - h_i)
        sigma1 = np.sum((Y - h_i)*X)

        theta0 = theta0 + eta/m*sigma0
        theta1 = theta1 + eta/m*sigma1
        print("Lan lap: {}\n\tsigma0={}\n\tsigma1={}\n\ttheta0={}\n\ttheta1={}".format(cnt, sigma0, sigma1, theta0, theta1))
    return [theta0, theta1]

if __name__ == "__main__":
    df = pd.read_csv("Housing_2019.csv", index_col=0)
    X = df.loc[:, ["lotsize", "bedrooms", "stories", "garagepl"]]
    Y = df.price
    print(X)
    print(Y)
    
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=1.0/3, random_state=7)
    lrm = LinearRegression()
    lrm.fit(X_train, Y_train)
    print("He so theta0: {}".format(lrm.intercept_))
    print("He so model:\n", lrm.coef_)

    Y_pred = lrm.predict(X_test)
    mse = mean_squared_error(Y_test, Y_pred)
    rmse = np.sqrt(mse)
    print("MSE: {}".format(mse))
    print("RMSE: {}".format(rmse))

    X = np.array([1, 2, 4])
    Y = np.array([2, 3, 6])

    theta_lr1 = LR1(X, Y, 0.2, 2, 0, 1)
    theta_lr2 = LR2(X, Y, 0.2, 2, 0, 1)
    print("He so LR1:\n", theta_lr1)
    print("He so LR2:\n", theta_lr2)

    X_test = np.array([0, 3, 5])
    Y_pred_lr1 = theta_lr1[0] + theta_lr1[1]*X_test
    Y_pred_lr2 = theta_lr2[0] + theta_lr2[1]*X_test
    print("Du doan LR1:\n", Y_pred_lr1)
    print("Du doan LR2:\n", Y_pred_lr2)

    #plt.axis([-10, 10, -10, 10])
    plt.plot(X, Y, "ro", color="blue")
    plt.plot(X_test, Y_pred_lr1, color="violet")
    plt.plot(X_test, Y_pred_lr2, color="green")
    plt.xlabel("Gia tri thuoc tinh X")
    plt.ylabel("Gia tri du doan Y")
    plt.show()