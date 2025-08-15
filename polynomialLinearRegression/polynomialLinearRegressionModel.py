import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score

df = pd.read_csv("Position_Salaries.csv")
X = df.iloc[:, 1:2].values
y = df.iloc[:, -1].values

#Simple Linear Regression
lin_reg = LinearRegression()
lin_reg.fit(X, y)

#Polynomial Regression
poly_reg = PolynomialFeatures(degree=4)
X_poly = poly_reg.fit_transform(X)
linTOpoly_reg = LinearRegression()
linTOpoly_reg.fit(X_poly, y)

plt.scatter(X, y, color='red')
plt.plot(X, lin_reg.predict(X), color='blue')
plt.title("Linear Regression")
plt.xlabel("Positions")
plt.ylabel("Salary")
plt.show()

plt.scatter(X, y, color='red')
plt.plot(X, linTOpoly_reg.predict(X_poly), color='blue')
plt.title("Polynomial Regression")
plt.xlabel("Positions")
plt.ylabel("Salary")
plt.show()

#For higher resolution and smoother curve
X_grid = np.arange(min(X), max(X), 0.1)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X, y, color='red')
plt.plot(X_grid, linTOpoly_reg.predict(poly_reg.fit_transform(X_grid)), color='blue')
plt.title("Polynomial Regression (Smoother)")
plt.xlabel("Positions")
plt.ylabel("Salary")
plt.show()

#Predict salary in level of 6.5 with Linear Regression
pred_lin = lin_reg.predict([[6.5]]) #2D Array
print(pred_lin)

#Predict salary in level of 6.5 with Polynomial Regression
pred_poly = linTOpoly_reg.predict(poly_reg.fit_transform([[6.5]]))

print(pred_poly)

#Evaluating the Model Performance
print(r2_score(y_val, y_pred))
