"""
TODO
add Neural Noob implementation
"""
import numpy as np
import matplotlib.pyplot as plt

from scipy.stats import norm
x = np.linspace(-3,3, 100)
# the mean of the Gaussian is set to 0.
# (the Gaussian is centred on 0.)
# the width of the Gaussian sigma is set to 1.
plt.plot(x, norm.pdf(x, 0., 1.))
plt.xlabel('x')
plt.ylabel('pdf')
plt.show()

normal = np.random.normal
sigma = 1
x0 = normal(-1.5, sigma, 100)
x1 = normal(1.5, sigma, 100)
# labels:
y0 = np.zeros_like(x0)
y1 = np.ones_like(x1)

plt.xlim(-5,5)
plt.plot(x0, y0,'o')
plt.plot(x1, y1,'o')
plt.xlabel('x')
plt.ylabel('category')

# plt.hist(sample1,bins=50, range=(-5,5))
plt.clf()
plt.xlim(-5,5)
plt.hist(x0,bins=50, range=(-5,5), alpha=0.5)
plt.hist(x1,bins=50, range=(-5,5), alpha=0.5)
plt.xlabel('x')
plt.ylabel('counts')
plt.show()

# define parameters
b = 0
w = 1

def sigmoid(x1):
    # z is a linear function of x1
    z = w*x1 + b
    return 1 / (1+np.exp(-z))

# create an array of evenly spaced values
linx = np.linspace(-5,5,51)
plt.plot(x0, np.zeros_like(x0),'o')
plt.plot(x1, np.ones_like(x1),'o')
plt.plot(linx, sigmoid(linx), color='red')
plt.xlabel('z')
plt.ylabel(r'$\sigma(z)$')
plt.show()

# create a 1D array containing 
# the values of x0 and x1:
x = np.concatenate((x0, x1))
# turn x into a 2D array with 1 value per line
# the first dimension indexes the examples, 
# and the second dimension contains the value
# for each example:
x = np.c_[x]
# create a 1D array with the targets 
# y0 and y1
y = np.concatenate((y0, y1))

from sklearn.linear_model import LogisticRegression

clf = LogisticRegression(solver='lbfgs').fit(x,y)
clf.predict_proba([[4]])

linx = np.c_[np.linspace(-5, 5, 100)]
# predict_proba takes an array of examples, 
# so a 2D array
prob = clf.predict_proba(linx)
# extract the second probability 
# (to be in category 1) for each example.
# we get a 2D array and 
# reshape it to a 1D array of size 100
prob = prob[:,1].reshape(len(linx))

# both linx and prob must be 1D
plt.plot(linx, prob, color='red')
plt.scatter(x0, y0)
plt.scatter(x1, y1)
plt.xlabel('x')
plt.ylabel('category probability')
plt.show()

from sklearn.neural_network import MLPClassifier

mlp = MLPClassifier(hidden_layer_sizes=(), 
                    solver='lbfgs')

mlp.fit(x,y)

prob_mlp = mlp.predict_proba(linx)
# take the second probability 
# (to be in category 1) for each example
# and reshape it to a 1D array of size 100
prob_mlp = prob_mlp[:,1].reshape(len(linx))
plt.plot(linx, prob, color='red', label='regression')
plt.plot(linx, prob_mlp, color='blue', label='MLP')
plt.scatter(x0, y0)
plt.scatter(x1, y1)
plt.legend()
plt.xlabel('x')
plt.ylabel('category probability')
plt.show()

