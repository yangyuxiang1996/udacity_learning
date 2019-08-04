import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, make_scorer
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV


sns.set_style('whitegrid')
sns.set_context('paper')


data = pd.read_csv('data.csv', header=None)
X = np.array(data.drop(2, axis=1))
y = np.array(data[2])

plt.figure()
plt.scatter(X[y == 1, 0], X[y == 1, 1], s=30, color='red', edgecolors='w')
plt.scatter(X[y == 0, 0], X[y == 0, 1], s=30, color='blue', edgecolors='w')
plt.show()

model = SVC(kernel='rbf', gamma=0.1, C=10)
model.fit(X, y)

y_pred = model.predict(X)

acc = accuracy_score(y_true=y, y_pred=y_pred)

print('the accuracy is : ', acc)

kf = KFold(n_splits=10, shuffle=True, random_state=1)
param = {'C': [i for i in range(1, 11)], 'gamma': [2**i for i in range(-5, 6)]}
classifier = SVC(kernel='rbf')
scorer = make_scorer(accuracy_score)

grid = GridSearchCV(estimator=classifier, param_grid=param, cv=kf, scoring=scorer)
grid.fit(X, y)

reg = grid.best_estimator_

print('the best C parameters is:{}'.format(grid.best_params_['C']))
print('\nthe best gamma parameters is:{}'.format(grid.best_params_['gamma']))

y_pred = reg.predict(X)
acc = accuracy_score(y_true=y, y_pred=y_pred)
print('the accuracy is:', acc)






