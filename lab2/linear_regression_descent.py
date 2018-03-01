#
# gradient descent for linear regression
# 

import numpy as np
import matplotlib.pyplot as plt
import sys

num_points1 = 1000
mean=[4, 6]
cov = [[1.5,2],[2,4]];

x,y = np.random.multivariate_normal(mean , cov, num_points1).T
print(x.shape, y.shape)
a,b = np.random.randn(2)
print(a,b)

def loss(a,b,x,y):
	return 1.0/x.shape[0] * np.sum((y - (a*x+b))**2)
	
def grad_step(a,b,x,y,step = 0.01):
	n = x.shape[0]
	grad_a = -2.0/n * np.dot(x, y - (a*x +b))
	grad_b = -2.0/n * np.sum(y - (a*x +b))
	#print('Grads\n', grad_a, '\n', grad_b)
	return a - step * grad_a, b - step * grad_b
	
def gradient_descent(x,y, epsilon = 0.001):
	a,b = np.random.randn(2)
	losses = []
	prev_loss = sys.maxsize
	curr_loss = 0
	steps = 0
	while np.abs(curr_loss - prev_loss) >  epsilon:
		losses += [loss(a,b,x,y)]
		a,b = grad_step(a,b,x,y, 0.01)
		steps += 1
		prev_loss = curr_loss
		curr_loss = loss(a,b,x,y)
	return a,b,losses, steps

xx = np.linspace(min(x), max(x))
a,b, losses, steps_count = gradient_descent(x,y)
print(steps_count, losses)
plt.figure(1)
plt.scatter(x,y,c='b')
plt.plot(xx, a*xx+b, 'r')	
plt.figure(2)
plt.plot(np.arange(len(losses)), losses)
plt.show()  
