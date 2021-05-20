from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier

import pandas as pd

if __name__ == "__main__":
    wineRed = pd.read_csv("winequality-red.csv", delimiter=';')
    print("Dataset columns:")
    print(wineRed.columns, '\n')
    print("Label value:")
    print(wineRed.quality.unique(), '\n')

    x = wineRed.drop("quality", axis=1)
    y = wineRed.quality # label
    print("Train data:")
    print(x, '\n')
    print("Test data")
    print(y, '\n')

    kf = KFold(
        n_splits = 60,
        shuffle = True,
        random_state = 7
    )
    train_size = 0
    test_size = 0
    cnt = 0
    for train_index, test_index in kf.split(x):
        train_size = train_size + len(train_index)
        test_size = test_size + len(test_index)
        cnt = cnt + 1
    print("Train size:\t{}".format(train_size/cnt))
    print("Test size:\t{}\n".format(test_size/cnt))

    cnt = 0
    acc_score_mean = 0
    for train_index, test_index in kf.split(x):
        x_train, y_train = x.iloc[train_index], y.iloc[train_index]
        x_test, y_test = x.iloc[test_index], y.iloc[test_index]

        gnb = GaussianNB()
        gnb.fit(x_train, y_train)

        y_pred = gnb.predict(x_test)
        cfm = confusion_matrix(y_test, y_pred)
        print("Iteration {}:".format(cnt))
        print("Confusion matrix:")
        print(cfm)
        acc_score = accuracy_score(y_test, y_pred)
        print("Accuracy score:\t{}\n".format(acc_score))
        acc_score_mean = acc_score_mean + acc_score
        cnt = cnt + 1
        '''
        Iteration 59:
        Confusion matrix:
        [[0 0 0 0]
        [3 4 0 1]
        [0 3 7 1]
        [0 1 3 3]]
        Accuracy score: 0.5384615384615384
        '''
    print("Mean accuracy score:\t{}\n".format(acc_score_mean/cnt))

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=1/3,
        random_state=7
    )

    dtc = DecisionTreeClassifier(
        max_depth=10,
        min_samples_leaf=3,
        random_state=7
    )
    dtc.fit(x_train, y_train)
    dtc_y_pred = dtc.predict(x_test)

    gnb = GaussianNB()
    gnb.fit(x_train, y_train)
    gnb_y_pred = gnb.predict(x_test)

    dtc_acc_score = accuracy_score(y_test, dtc_y_pred)
    gnb_acc_score = accuracy_score(y_test, gnb_y_pred)

    print("DecisionTreeClassifier accuracy score:\t{}".format(dtc_acc_score))
    print("GaussianNB accuracy score:\t{}".format(gnb_acc_score))