import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from gplearn.genetic import SymbolicRegressor
from sklearn.model_selection import train_test_split
from sympy import *


def f(x):
    return x * x * x * x - 4 * x * x * x + x * x - 5 * x + 1

converter = {
    'sub': lambda x, y : x - y,
    'div': lambda x, y : x/y,
    'mul': lambda x, y : x*y,
    'add': lambda x, y : x + y,
    'sin': lambda x    : sin(x),
    'cos': lambda x    : cos(x),
}

nsample = 1000
sig = 0.2
x = np.linspace(-10, 10, nsample)
y = np.array([f(x_i) for x_i in x])
df = pd.DataFrame()
df['x']=x
df['y']=y


X = df[['x']]
y = df['y']
y_true = y
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)

# First Test
function_set = ['add', 'sub', 'mul', 'div','cos','sin','neg','inv']
est_gp = SymbolicRegressor(population_size=1000,function_set=function_set,
                           generations=40, stopping_criteria=0.01,
                           p_crossover=0, p_subtree_mutation=0.1,
                           p_hoist_mutation=0.05, p_point_mutation=0.1,
                           max_samples=0.9, verbose=1,
                           parsimony_coefficient=0.01, random_state=0,
                          feature_names=X_train.columns)

est_gp.fit(X_train, y_train)
print('R2:',est_gp.score(X_test,y_test))
next_e = sympify((est_gp._program), locals=converter)
next_e

y_predict = est_gp.predict(X_test)

print(y_predict)

X_test_np_b = np.array(X_test)
y_test_np = np.array(y_test)
X_test_np = X_test_np_b.T[0]
y_predict_np = np.array(y_predict)


true_values = np.array(list(zip(X_test_np,y_test_np)))
predicted_values = np.array(list(zip(X_test_np,y_predict_np)))
true_values = np.array(sorted(true_values,key=lambda x : x[0]))
predicted_values = np.array(sorted(predicted_values,key=lambda x : x[0]))
X_true_values = [x[0] for x in true_values]
Y_true_values = [x[1] for x in true_values]
Y_predicted_values = [x[1] for x in predicted_values]
plt.plot(X_true_values,Y_true_values,'r')
plt.plot(X_true_values,Y_predicted_values,'b')
plt.show()

