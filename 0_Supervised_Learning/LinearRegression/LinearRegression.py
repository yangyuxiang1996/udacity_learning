import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression


data = pd.read_csv('bmi_and_life_expectancy.csv')
print(data.head())

bmi_data = data[['BMI']]
life_true = data[['Life expectancy']]

plt.figure()
plt.scatter(bmi_data, life_true, color='blue')
# plt.xticks(np.arange(bmi_life_data.min(), bmi_life_data.max(), step=5))
plt.xlabel('Life expectancy')
plt.ylabel('BMI')

bmi_life_model = LinearRegression()
bmi_life_model.fit(bmi_data, life_true)


bmi_life_pre = np.array(21.07931).reshape(1, -1)
laos_life_exp = bmi_life_model.predict(bmi_life_pre)
print('the expected life expectancy for a BMI value of 21.07931 is {}'.format(laos_life_exp))

X = np.arange(bmi_data.min().values[0], bmi_data.max().values[0], step=0.5).reshape(-1, 1)
plt.plot(X, bmi_life_model.predict(X), color='red')
plt.show()


# # TODO: Add import statements
# import pandas as pd
# import numpy as np
# from sklearn.linear_model import LinearRegression
#
# # Assign the dataframe to this variable.
# # TODO: Load the data
# bmi_life_data = pd.read_csv("bmi_and_life_expectancy.csv")
#
# # Make and fit the linear regression model
# #TODO: Fit the model and Assign it to bmi_life_model
# bmi_life_model = LinearRegression()
# bmi_life_model.fit(bmi_life_data[['BMI']], bmi_life_data[['Life expectancy']])
#
# # Mak a prediction using the model
# # TODO: Predict life expectancy for a BMI value of 21.07931
# laos_life_exp = bmi_life_model.predict(np.array(21.07931).reshape(-1, 1))
# print('the expected life expectancy for a BMI value of 21.07931 is ', laos_life_exp)