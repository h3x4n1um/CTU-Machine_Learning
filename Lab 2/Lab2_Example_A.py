from sklearn import metrics
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

if __name__ == "__main__":
    iris_dt = load_iris()
    iris_dt.data[1:5]
    iris_dt.target[1:5]

    x_train, x_test, y_train, y_test = train_test_split(iris_dt.data, iris_dt.target, test_size=1/3.0, random_state=5)

    clf_gini = DecisionTreeClassifier(
        criterion = "gini",
        random_state = 100,
        max_depth = 3,
        min_samples_leaf = 5
    )
    clf_gini.fit(x_train, y_train)
    y_pred = clf_gini.predict(x_test)
    print(y_pred)
    print(y_test)
    print(clf_gini.predict([[4, 4, 3, 3]]))
    print("Accuracy is ", metrics.accuracy_score(y_test, y_pred)*100)
    print(metrics.confusion_matrix(y_test, y_pred, labels=[2, 0, 1]))
