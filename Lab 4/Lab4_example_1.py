import numpy as np
import matplotlib.pyplot as plt

def LR1(X, Y, eta, loop_cnt, theta0, theta1):
    m = len(X)
    for k in range(loop_cnt):
        print("Lan lap:\t{}".format(k))
        for i in range(m):
            h_i = theta0 + theta1*X[i]

            theta0 = theta0 + eta*(Y[i]-h_i)*1
            theta1 = theta1 + eta*(Y[i]-h_i)*X[i]

            print("Phan tu thu {} y={} h={} gia tri theta0={} theta1={}".format(i, Y[i], h_i, round(theta0, 3), round(theta1, 3)))
    return [round(theta0, 3), round(theta1, 3)]

if __name__ == "__main__":
    X = np.array([1, 2, 4])
    Y = np.array([2, 3, 6])
    
    plt.axis([0, 10, 0, 10])
    plt.plot(X, Y, "ro", color="blue")
    plt.xlabel("Gia tri thuoc tinh X")
    plt.ylabel("Gia tri thuoc tinh Y")
    plt.show()

    theta = LR1(X, Y, 0.2, 1, 0, 1)
    theta2 = LR1(X, Y, 0.2, 2, 0, 1)
    print(theta)
    print(theta2)

    X_test = np.array([1, 6])
    Y1_pred = theta[0] + theta[1]*X_test
    Y2_pred = theta2[0] + theta2[1]*X_test

    plt.axis([0, 10, 0, 10])
    plt.plot(X, Y, "ro", color="blue")
    plt.plot(X_test, Y1_pred, color="violet")
    plt.plot(X_test, Y2_pred, color="green")
    plt.xlabel("Gia tri thuoc tinh X")
    plt.ylabel("Gia tri du doan Y")
    plt.show()

    theta = LR1(X, Y, 0.1, 1, 0, 1)
    theta2 = LR1(X, Y, 0.1, 2, 0, 1)
    print(theta)
    print(theta2)

    Y1_pred = theta[0] + theta[1]*X_test
    Y2_pred = theta2[0] + theta2[1]*X_test

    plt.axis([0, 10, 0, 10])
    plt.plot(X, Y, "ro", color="blue")
    plt.plot(X_test, Y1_pred, color="violet")
    plt.plot(X_test, Y2_pred, color="green")
    plt.xlabel("Gia tri thuoc tinh X")
    plt.ylabel("Gia tri du doan Y")
    plt.show()

    theta = LR1(X, Y, 0.2, 2, 0, 1)
    X_test = np.array([0, 3, 5])
    Y_pred = theta[0] + theta[1]*X_test
    print(Y_pred)