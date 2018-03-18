# Stanisław Wilczyński project1
import sys

import matplotlib.pyplot as plt
import numpy as np

n = 1000
d = 10


def loss(a, x, y):
    xb = np.dot(x, a)
    return np.sum(np.log(1 + np.exp(xb)) - np.multiply(y, xb))


def generate_data(n, d, ro, option, debug=False):
    mean = np.zeros(d)
    if option == 'independent':
        cov = np.identity(d)
        if debug:
            print('Mean and covariance {} \n {}'.format(mean, cov))
        return np.random.multivariate_normal(mean, cov, n)
    if option == 'same_corr':
        cov = np.ones((d, d)) * ro + np.identity(d) * (1 - ro)
        if debug:
            print('Mean and covariance {} \n {}'.format(mean, cov))
        return np.random.multivariate_normal(mean, cov, n)
    if option == 'auto_corr':
        cov = [[ro ** np.abs(i - j) for i in range(d)] for j in range(d)]
        if debug:
            print('Mean and covariance {} \n {}'.format(mean, cov))
        return np.random.multivariate_normal(mean, cov, n)

    raise Exception('Wrong option')


def generate_response(X, beta, debug=False):
    p = np.exp(np.dot(X, beta)) / (1 + np.exp(np.dot(X, beta)))
    if debug:
        print('Probability\n', p)
    return np.random.binomial(1, p).reshape(-1, 1)


# proximal for lambda penalty - as in the lecture
def proximal(a, step, lam):
    return np.multiply(np.sign(a), np.maximum((np.abs(a) - lam * step), np.zeros((d, 1))))


# gradient step for logistic regression model - as in project
def grad_step(a, x, y, step=0.01):
    p = np.exp(np.dot(x, a)) / (1 + np.exp(np.dot(x, a))).reshape(n, 1)
    grad_a = np.dot(x.T, (p - y))
    return a - step * grad_a


# stochastic gradient step - calculate gradient with respect only to loss connected with randomly chosen x_j
def stochastic_grad_step(a, x, y, step=0.01):
    j = np.random.randint(n)
    p = np.exp(np.dot(x[j, :], a)) / (1 + np.exp(np.dot(x[j, :], a))).ravel()[0]
    grad_a = (p - y[j]) * x[j, :]
    return a - step * grad_a.reshape(-1, 1)


# gradient descent algorithm
def gradient_descent(x, y, epsilon=0.0000000001, step=0.01, lam=0.01, prox=False, stochastic=False):
    a = np.random.randn(d).reshape(d, 1)
    losses = []
    prev_loss = sys.maxsize
    curr_loss = 0
    steps = 0
    while np.abs(curr_loss - prev_loss) > epsilon and steps < 10 ** 4:
        losses += [loss(a, x, y)]
        if stochastic:
            a = stochastic_grad_step(a, x, y, step)
        else:
            a = grad_step(a, x, y, step)
        if prox:
            a = proximal(a, step, lam)
        steps += 1
        prev_loss = curr_loss
        curr_loss = loss(a, x, y)
    return a, losses, steps


# saga algorithm
def saga(x, y, epsilon=0.00000001, step=0.01, prox=False):
    a = np.zeros(d).reshape(d, 1)
    p = np.exp(np.dot(x, a)) / (1 + np.exp(np.dot(x, a))).reshape(n, 1)
    M = np.multiply(p - y, x)
    avg = np.mean(M, axis=0)
    losses = []
    prev_loss = sys.maxsize
    curr_loss = 0
    steps = 0
    while np.abs(curr_loss - prev_loss) > epsilon and steps < 10 ** 4:
        losses += [loss(a, x, y)]
        ####saga step
        j = np.random.randint(n)
        old_deriv = np.array(M[j, :])  # very important copy
        p = np.exp(np.dot(x[j, :], a)) / (1 + np.exp(np.dot(x[j, :], a))).ravel()[0]
        new_deriv = (p - y[j]) * x[j, :]
        M[j, :] = new_deriv
        a -= step * ((new_deriv - old_deriv + avg).reshape(-1, 1))
        avg += (new_deriv - old_deriv) / n
        if prox:
            a = proximal(a, step, lam=0.01)
        ####
        steps += 1
        prev_loss = curr_loss
        curr_loss = loss(a, x, y)
    return a, losses, steps


# svrg algorithm
def svrg(x, y, epsilon=0.00001, step=0.01, lam=0.01, prox=False):
    pass


beta = np.random.uniform(-1, 2, d).reshape(-1, 1)  # very important reshape

###mostly zeros beta
# ind = np.random.permutation(np.arange(d))[:-d//8]
# beta[ind] = 0
####
x = generate_data(n, d, 0.9, 'independent')
y = generate_response(x, beta)

start_ind = 2
a, losses, steps_count = gradient_descent(x, y)
print('Number of nonzero coeff: {}'.format(np.sum(a != 0)))
print('Squared error for betas: {}'.format(np.sum((a - beta) ** 2)))
print('Steps count for GD: {}\nThe loss was {}'.format(steps_count, losses[-1]))
plt.figure(1)
plt.plot(np.arange(len(losses[start_ind:])), losses[start_ind:], label='Loss for GD')

a, losses, steps_count = gradient_descent(x, y, step=0.02, stochastic=True)
print('Number of nonzero coeff: {}'.format(np.sum(a != 0)))
print('Squared error for betas: {}'.format(np.sum((a - beta) ** 2)))
print('Steps count for SGD: {}\nThe loss was {}'.format(steps_count, losses[-1]))
plt.figure(1)
plt.plot(np.arange(len(losses[start_ind:])), losses[start_ind:], label='Loss for SGD')

a, losses, steps_count = saga(x, y, step=0.05)
print('Number of nonzero coeff: {}'.format(np.sum(a != 0)))
print('Squared error for betas: {}'.format(np.sum((a - beta) ** 2)))
print('Steps count for SAGA: {}\nThe loss was {}'.format(steps_count, losses[-1]))
plt.figure(1)
plt.plot(np.arange(len(losses[start_ind:])), losses[start_ind:], label='Loss for SAGA')

plt.legend()
plt.show()
