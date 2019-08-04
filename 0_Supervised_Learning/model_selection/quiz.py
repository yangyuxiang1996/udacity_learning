import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.model_selection import learning_curve

data = pd.read_csv('data.csv')
print(data.head())
X = np.array(data[['x1', 'x2']])
y = np.array(data['y'])

np.random.seed(55)


def randomize(X, Y):
    permutation = np.random.permutation(Y.shape[0])
    X2 = X[permutation, :]
    Y2 = Y[permutation]
    return X2, Y2


def draw_learning_curve(X, y, estimator, num_trainings, title):
    train_sizes, train_scores, test_scores = learning_curve(estimator, X, y, cv=3, n_jobs=1,
                                                            train_sizes=np.linspace(.1, 1.0, num_trainings))
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)

    plt.figure()
    plt.grid()
    plt.title(title)
    # plt.title('Learning Curves')
    plt.xlabel('Training Size')
    plt.ylabel('score')

    plt.plot(train_scores_mean, 'o-', color='g', label='Training score')
    plt.plot(test_scores_mean, 'o-', color='r', label='Testing score')

    plt.legend(loc='best')
    plt.show()


X2, Y2 = randomize(X, y)

estimator = LogisticRegression(solver='liblinear')
draw_learning_curve(X2, Y2, estimator, 5, title='LR Learning Curves')

estimator1 = GradientBoostingClassifier()
draw_learning_curve(X2, Y2, estimator1, 5, title='GB Learning Curves')

estimator2 = SVC(kernel='rbf', gamma=1000)
draw_learning_curve(X2, Y2, estimator2, 5, title='SVM Learning Curves')

