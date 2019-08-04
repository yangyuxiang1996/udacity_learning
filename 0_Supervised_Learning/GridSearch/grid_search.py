import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, make_scorer
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier
import random


def load_pts(csv_name):
    data = np.asarray(pd.read_csv(csv_name, header=None))

    X = data[:, 0:2]
    y = data[:, 2]

    plt.scatter(X[np.argwhere(y == 0).flatten(), 0], X[np.argwhere(y == 0).flatten(), 1], c='blue',
                edgecolor='k', s=50)
    plt.scatter(X[np.argwhere(y == 1).flatten(), 0], X[np.argwhere(y == 1).flatten(), 1], c='red',
                edgecolor='k', s=50)
    plt.xlim(-2.05, 2.05)
    plt.ylim(-2.05, 2.05)
    plt.grid(False)
    plt.tick_params(axis='x', which='both', bottom=False, top=False)

    return X, y


X, y = load_pts('data.csv')
plt.show()

random.seed(42)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

clf = DecisionTreeClassifier(random_state=42)

clf.fit(X_train, y_train)

train_predictions = clf.predict(X_train)
test_predictions = clf.predict(X_test)


def plot_model(X, y, clf):

    plt.scatter(X[np.argwhere(y == 0).flatten(), 0], X[np.argwhere(y == 0).flatten(), 1], c='blue',
                edgecolor='k', s=50)
    plt.scatter(X[np.argwhere(y == 1).flatten(), 0], X[np.argwhere(y == 1).flatten(), 1], c='red',
                edgecolor='k', s=50)
    plt.xlim(-2.05, 2.05)
    plt.ylim(-2.05, 2.05)
    plt.grid(False)
    plt.tick_params(axis='x', which='both', bottom=False, top=False)

    r = np.linspace(-2.1, 2.1, 300)
    s, t = np.meshgrid(r, r)
    s = np.reshape(s, (np.size(s), 1))
    t = np.reshape(t, (np.size(t), 1))
    h = np.concatenate((s, t), 1)

    z = clf.predict(h)

    s = s.reshape((np.size(r), np.size(r)))
    t = t.reshape((np.size(r), np.size(r)))
    z = z.reshape((np.size(r), np.size(r)))

    plt.contourf(s, t, z, colors=['blue', 'red'], alpha=0.2, levels=range(-1, 2))
    if len(np.unique(z)) > 1:
        plt.contour(s, t, z, colors='k', linewidths=2)
    plt.show()


plot_model(X, y, clf)
print('The Training F1 Score is', f1_score(train_predictions, y_train))
print('The Testing F1 Score is', f1_score(test_predictions, y_test))

clf = DecisionTreeClassifier(random_state=42)

parameters = {'max_depth': [2, 4, 6, 8, 10], 'min_samples_leaf': [2, 4, 6, 8, 10],
              'min_samples_split': [2, 4, 6, 8, 10]}

scorer = make_scorer(f1_score)

grid_obj = GridSearchCV(clf, parameters, scoring=scorer, cv=3)

grid_fit = grid_obj.fit(X_train, y_train)

best_clf = grid_fit.best_estimator_

best_clf.fit(X_train, y_train)

best_train_predictions = best_clf.predict(X_train)
best_test_predictions = best_clf.predict(X_test)
print('The training F1 Score is', f1_score(best_train_predictions, y_train))
print('The testing F1 Score is', f1_score(best_test_predictions, y_test))

plot_model(X, y, best_clf)

print(best_clf)


