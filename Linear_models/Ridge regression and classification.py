# Setting the regularization parameter: leave-one-out Cross-Validation
import numpy as np
from sklearn import linear_model
reg = linear_model.RidgeCV(alphas = np.logspace(-6, 6, 13))
reg.fit([[0, 0], [0, 0], [1, 1]], [0, .1, 1])
reg.alpha_
