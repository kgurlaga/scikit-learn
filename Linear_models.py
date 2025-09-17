# 1.1.1. Ordinary Least Squeres
from sklearn import linear_model
import numpy as np
reg = linear_model.LinearRegression()
reg.fit([[0, 0], [1, 1], [2, 2]], [0, 1, 2])
print(reg.coef_)
intercept = reg.intercept_
intercept = 0 if np.isclose(intercept, 0) else intercept
print(intercept)

# Ordinary Least Squares and Ridge Regression