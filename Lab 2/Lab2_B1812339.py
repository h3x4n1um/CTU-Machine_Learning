from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

import pandas as pd

if __name__ == "__main__":
    df = pd.read_csv("winequality-white.csv", delimiter=';')

    print("Tong so phan tu\t\t\t\t", len(df))
    print("So gia tri nhan khac nhau\t\t", len(df["quality"].unique()))

    x = df.drop("quality", 1)
    y = df["quality"]
    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.2,
        random_state=5)
    print("So phan tu trong tap test\t\t", len(y_test))
    print("So phan tu trong tap train\t\t", len(y_train))

    model = DecisionTreeClassifier(
        criterion = "entropy",
        random_state = 7
    )
    model.fit(x_train, y_train)

    y_pred = model.predict(x_test)
    print("Do chinh xac tong the\t\t\t", metrics.accuracy_score(y_test, y_pred))
    print("Do chinh xac tung lop\n", metrics.confusion_matrix(y_test, y_pred))

    print("Do chinh xac tong the 10 phan tu dau\t", metrics.accuracy_score(y_test[:10], y_pred[:10]))
    print("Do chinh xac tung lop 10 phan tu dau\n", metrics.confusion_matrix(y_test[:10], y_pred[:10]))



    X2 = pd.DataFrame(
        data=[
            [180, 15, 0],
            [167, 42, 1],
            [136, 35, 1],
            [174, 15, 0],
            [141, 28, 1]
        ],
        columns=["Chieu cao", "Do dai mai toc", "Giong noi"])
    Y2 = pd.DataFrame(data=["Nam", "Nu", "Nu", "Nam", "Nu"], columns=["Nhan"])

    model2 = DecisionTreeClassifier(
        criterion = "entropy",
        random_state = 7
    )
    model2.fit(X2, Y2)

    print("Ket qua du doan [130, 38, 1]\t\t", model2.predict([[130, 38, 1]]))
