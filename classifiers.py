import matplotlib.pyplot as plt

from sklearn.metrics import mean_absolute_error, precision_recall_curve
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score,KFold, cross_validate
from sklearn import tree


"""
Jia Wu, Khadija Jallow, & Tadius Frank 
May 3, 2022

Classifiers contains evaluation metrics for decision tree and 
linear regression

"""

num_split = 10 

# set number of folds for k fold  
def setK(num):
    num_split = num 

# prints error accuracy and test case i 
def accuracy(Y_predict, Y_test, i): 

    acc = sum(Y_predict == Y_test) / len(Y_test)  # accuray
    print("accuracy: \t" + str(acc) + "\t" + str(i))
    
# PRC curve not necessary because our data is evenly distributed 
def predicsionRecallCurve(Y_test, y_score): 
    # PRC 
    precision, recall, threshold = precision_recall_curve(Y_test, y_score)
    
    #create precision recall curve
    fig, ax = plt.subplots()
    ax.plot(recall, precision, color='purple')

    #add axis labels to plot
    ax.set_title('Precision-Recall Curve')
    ax.set_ylabel('Precision')
    ax.set_xlabel('Recall')

    #display plot
    plt.show()

def checkFit(clf, X, y):
    kf=KFold(n_splits=num_split)
    mae_train = []
    mae_test = []

    for train_index, test_index in kf.split(y):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        clf.fit(X_train, y_train)

        y_train_pred = clf.predict(X_train)
        y_test_pred = clf.predict(X_test)

        mae_train.append(mean_absolute_error(y_train, y_train_pred))
        mae_test.append(mean_absolute_error(y_test, y_test_pred))
    
    folds = range(1, kf.get_n_splits() + 1)
    plt.plot(folds, mae_train, 'o-', color='green', label='train')
    plt.plot(folds, mae_test, 'o-', color='red', label='test')
    plt.legend()
    plt.grid()
    plt.xlabel('Number of fold')
    plt.ylabel('Mean Absolute Error')
    plt.show()

# cross validation with train, score time, and scores 
def kfold_all(clf, X, Y, i):
    kf=KFold(n_splits=num_split)
    #report i parameter 
    print(i)
    # evaluate model, fit time, score time, test time 
    cv = cross_validate(clf, X, Y, cv=kf, n_jobs=1)
    # report performance
    for key in cv.keys():
        print(key)
        for value in cv[key]:
            print(value)

# 
def kfold_score(clf, X, Y, i):
    kf=KFold(n_splits=num_split)
    #report i parameter 
    print(i)
    # evaluate model, fit time, score time, test time 
    scores = cross_val_score(clf, X, Y, cv=kf, n_jobs=1)
    # report performance
    for score in scores:
        print(score)

# test logistic regression on different lambda and regularizers 
def testLogisticReg(X_train, Y_train, x, y):
    
    # where C is inverse of lambda i.e. if  λ=10 == C=0.1.
    C = [10, 2, 1.5, 1, .5, .1, .001]
    # L1 is Lasso regression and L2 is Ridge regression 
    regularizer = ["l1", "l2"]

    for l in regularizer: 
        print(l)
        for c in C:
            clf = LogisticRegression(C=c, penalty=l, solver='liblinear')
            clf.fit(X_train,Y_train)
            kfold_score(clf, x, y, c)

# test for different stopping criteria and pruning methods 
def testDT(X_train, Y_train, x, y): 
    criterion = ["gini", "entropy"]
    for c in criterion: 
        print(c)
        for depth in range(100, 1001, 100): 
            clf = DecisionTreeClassifier(criterion = c, random_state = 100, max_depth=depth, min_samples_leaf=50)
            clf = clf.fit(X_train, Y_train)

            kfold_score(clf, x, y, depth)
        # minimum number of samples required to be a leaf node
        for num_sample_a_leaf in range (10, 81, 10): 
            clf = DecisionTreeClassifier(criterion = c, random_state = 100, max_depth=200, min_samples_leaf=num_sample_a_leaf)
            clf = clf.fit(X_train, Y_train)

            kfold_score(clf, x, y, num_sample_a_leaf)
    
# plot the figure 
def plotDT(clf, features, label): 
    fig = plt.figure(figsize=(12,12))
    tree.plot_tree(clf, feature_names = features, class_names = label, filled =True, fontsize=10)
    fig.savefig("decision_tree.png") 
