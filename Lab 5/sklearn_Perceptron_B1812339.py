from sklearn.datasets import load_iris
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

if __name__ == "__main__":
    df = load_iris()
    x_train, x_test, y_train, y_test = train_test_split(df.data, df.target, test_size=1/3, random_state=7)
    max_iter_arr = [5, 50, 100, 1000]
    eta0_arr = [0.002, 0.02, 0.2]
    for max_iter in max_iter_arr:
        for eta0 in eta0_arr:
            print("max_iter:\t{}\neta0:\t\t{}".format(max_iter, eta0))
            pct = Perceptron(
                max_iter=max_iter,
                eta0=eta0,
                random_state=5
            )
            pct.fit(x_train, y_train)
            print("coef_:\n{}".format(pct.coef_))
            print("intercept_:\t{}".format(pct.intercept_))

            y_pred = pct.predict(x_test)
            print("accuracy_score:\t{}\n".format(accuracy_score(y_test, y_pred)))