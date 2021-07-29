from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
import pandas as pd

algos = []
def baseline(X_train,X_test,y_train,y_test):

    clf = SVC(kernel = "linear")
    clf.fit(X_train,y_train)
    algos.append(["SVM Linear",clf.score(X_test,y_test)])

    clf = SVC()
    clf.fit(X_train,y_train)
    algos.append(["SVM RBF",clf.score(X_test,y_test)])

    clf = KNeighborsClassifier()
    clf.fit(X_train,y_train)
    algos.append(["KNN",clf.score(X_test,y_test)])

    clf = DecisionTreeClassifier()
    clf.fit(X_train,y_train)
    algos.append(["Decision Tree",clf.score(X_test,y_test)])

    clf = RandomForestClassifier()
    clf.fit(X_train,y_train)
    algos.append(["Random Forest",clf.score(X_test,y_test)])

    clf = AdaBoostClassifier()
    clf.fit(X_train,y_train)
    algos.append(["AdaBoost",clf.score(X_test,y_test)])

    clf = GaussianNB()
    clf.fit(X_train,y_train)
    algos.append(["GaussianNB",clf.score(X_test,y_test)])

    clf = SGDClassifier()
    clf.fit(X_train,y_train)
    algos.append(["SGD",clf.score(X_test,y_test)])

    clf = LogisticRegression()
    clf.fit(X_train,y_train)
    algos.append(["LogisticRegression",clf.score(X_test,y_test)])

    clf = GradientBoostingClassifier()
    clf.fit(X_train,y_train)
    algos.append(["Gradient Boosting",clf.score(X_test,y_test)])

    clf = MLPClassifier()
    clf.fit(X_train,y_train)
    algos.append(["MLP",clf.score(X_test,y_test)])

    clf = GaussianProcessClassifier()
    clf.fit(X_train,y_train)
    algos.append(["Gaussian Process",clf.score(X_test,y_test)])

    algo_results_df = pd.DataFrame(algos,columns = ["Algorithm","Score"])
    algo_results_df.sort_values("Score",ascending=False,ignore_index=True, inplace= True)
    return algo_results_df