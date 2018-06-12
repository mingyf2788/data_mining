import numpy as np
import pandas as pd
# data = load_exdata('ex1data2.txt');
# data = np.array(data, np.int64)

# x = data[:, (0, 1)].reshape((-1, 2))
# y = data[:, 2].reshape((-1, 1))
# m = y.shape[0]
# print len(x)
# print len(x[0])

data = pd.read_csv('train1.csv')
x = data.iloc[:, :-1].values
y = data.iloc[:, 128].values
m = y.shape[0]


def featureNormalize(X):
    X_norm = X;
    mu = np.zeros((1, X.shape[1]))
    sigma = np.zeros((1, X.shape[1]))
    for i in range(X.shape[1]):
        mu[0, i] = np.mean(X[:, i])
        sigma[0, i] = np.std(X[:, i])
    X_norm = (X - mu) / sigma
    return X_norm, mu, sigma


def computeCost(X, y, theta):
    m = y.shape[0]
    print "in compute: "
    print len(theta)
    print len(theta[0])
    C = X.dot(theta) - y
    J2 = (np.sum((X.dot(theta) - y)**2)) / (2*m)
    # J2 = (C.T.dot(C)) / (2 * m)
    return J2


def gradientDescent(X, y, theta, alpha, num_iters):
    m = y.shape[0]
    J_history = np.zeros((num_iters, 1))
    print "gradientdescent: "
    print len(theta)
    print len(theta[0])
    for iter in range(num_iters):
        # theta = theta - (alpha / m) * (X.T.dot(X.dot(theta) - y))
        theta = theta - (alpha / m) * np.sum(X.T * (X.dot(theta) - y))
        J_history[iter] = computeCost(X, y, theta)
    print "gradientdescent: "
    print len(theta)
    print len(theta[0])
    return J_history, theta


iterations = 10000
alpha = 0.01
m = y.shape[0]
x, mu, sigma = featureNormalize(x)

X = np.hstack([x, np.ones((x.shape[0], 1))])
theta = np.zeros((129, 1))

j = computeCost(X, y, theta)
J_history, theta = gradientDescent(X, y, theta, alpha, iterations)

print('Theta found by gradient descent', theta)