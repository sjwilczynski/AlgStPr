#
# project first try
# 

import numpy as np
import matplotlib.pyplot as plt
import sys

n=1000
d=10


def loss(a,x,y):
	xb = np.dot(x,a)
	#print ('Shapes:', xb.shape, np.log(1+np.exp(xb)).shape, np.multiply(y, xb).shape)
	return np.sum(np.log(1+np.exp(xb)) - np.multiply(y, xb))
	
def ith_loss(a,x,y,i):
	xbi = np.dot(x[i,:],a)
	ith = (np.log(1+np.exp(xbi)) - y[i]*xbi)[0]
	#print( 'Ok if equal: {} {}'.format(ith, ((np.log(1+np.exp(np.dot(x,a))) - np.multiply(y, np.dot(x,a))).ravel())[i]))
	return ith

def generate_data(n,d, ro, option, debug=False):
	mean = np.zeros(d)
	if option == 'independent':
		cov = np.identity(d)
		if debug:
			print('Mean and covaraince {} \n {}'.format(mean, cov))
		return np.random.multivariate_normal(mean, cov, n)
	if option == 'same_corr':
		cov = np.ones((d,d))*ro + np.identity(d) * (1-ro)
		if debug:
			print('Mean and covaraince {} \n {}'.format(mean, cov))
		return np.random.multivariate_normal(mean, cov, n)
	if option == 'auto_corr':
		cov = [[ro**np.abs(i-j) for i in range(d)] for j in range(d)]
		if debug:
			print('Mean and covaraince {} \n {}'.format(mean, cov))
		return np.random.multivariate_normal(mean, cov, n)
	
	raise Exception('Wrong option')
		
def generate_response(X,beta, debug = False):
	p = np.exp(np.dot(X,beta))/(1+np.exp(np.dot(X,beta)))
	if debug:
		print('Probability\n', p)
	return np.random.binomial(1, p).reshape(-1,1)
	
	
#proximal for lambda penalty - as in the lecture	
def proximal(a, step, lam):
	#print(a.shape, np.multiply(np.sign(a), np.maximum((np.abs(a) -  lam*step), np.zeros((d,1)))).shape)
	return np.multiply(np.sign(a), np.maximum((np.abs(a) -  lam*step), np.zeros((d,1))))
	
def proximal_grad_step(a,x,y,step = 0.01, lam = 0.001):
	return proximal(grad_step(a,x,y,step), step, lam) 

#gradient for logistic regression model - as in project	
def grad_step(a,x,y,step = 0.01):
	p = np.exp(np.dot(x,a))/(1+np.exp(np.dot(x,a))).reshape(n,1)
	grad_a = np.dot(x.T,(p - y))
	return a - step * grad_a
	
def gradient_descent(x,y, epsilon = 0.0000000001):
	a = np.random.randn(d).reshape(d,1)
	losses = []
	prev_loss = sys.maxsize
	curr_loss = 0
	steps = 0
	while np.abs(curr_loss - prev_loss) >  epsilon and steps < 10**4:
		losses += [loss(a,x,y)]
		gk = ith_loss(a,x,y,10)
		a = grad_step(a,x,y, 0.01)
		steps += 1
		prev_loss = curr_loss
		curr_loss = loss(a,x,y)
	return a,losses, steps
	
def proximal_gradient_descent(x,y, epsilon = 0.001, lam = 0):
	a = np.random.randn(d).reshape(d,1)
	losses = []
	prev_loss = sys.maxsize
	curr_loss = 0
	steps = 0
	while np.abs(curr_loss - prev_loss) >  epsilon and steps < 10**5:
		losses += [loss(a,x,y)]
		a = proximal_grad_step(a,x,y, 0.01, lam)
		steps += 1
		prev_loss = curr_loss
		curr_loss = loss(a,x,y)
	return a,losses, steps
	

	

def saga(x,y, epsilon = 0.00000001, step = 0.01, prox = False):
	a = np.zeros(d).reshape(d,1)
	p = np.exp(np.dot(x,a))/(1+np.exp(np.dot(x,a))).reshape(n,1)
	M = np.multiply(p-y, x)
	avg = np.mean(M, axis=0)
	losses = []
	prev_loss = sys.maxsize
	curr_loss = 0
	steps = 0
	while np.abs(curr_loss - prev_loss) >  epsilon and steps < 10**5:
		losses += [loss(a,x,y)]
		
		####saga step
		j = np.random.randint(n)
		old_deriv = np.array(M[j,:]) #f**** copy
		p = np.exp(np.dot(x[j,:],a))/(1+np.exp(np.dot(x[j,:],a))).ravel()[0]
		new_deriv = (p - y[j])*x[j,:]
		M[j,:] = new_deriv
		a -= step * ((new_deriv - old_deriv + avg).reshape(-1,1))
		avg += (new_deriv - old_deriv)/n
		if prox:
			a = proximal(a, step, lam=0.01)
		####
		steps += 1
		prev_loss = curr_loss
		curr_loss = loss(a,x,y)
	return a,losses, steps


beta = np.random.uniform(-1,2,d).reshape(-1,1) # f*** shape

###mostly zeros beta
#ind = np.random.permutation(np.arange(d))[:-d//8]
#beta[ind] = 0
####
x = generate_data(n,d,0.9,'independent')
y = generate_response(x,beta)
print(beta.shape, x.shape, y.shape)

start_ind = 0
a,losses,steps_count = gradient_descent(x,y)
print('Number of nonzero coeff: {}'.format(np.sum(a!=0)))
print('Squared error for betas: {}'.format(np.sum((a-beta)**2)))
print('Steps count for GD: {}\nThe loss was {}'.format(steps_count, losses[-1]))
plt.figure(1)
plt.plot(np.arange(len(losses[start_ind:])), losses[start_ind:], label='Loss for GD')
#a,losses,steps_count = proximal_gradient_descent(x,y, lam = 10.5)
#print('Number of nonzero coeff: {}'.format(np.sum(a!=0)))
#print('Steps count for PGD: {}\nThe loss was {}'.format(steps_count, losses[-1]))
#plt.figure(1)
#plt.plot(np.arange(len(losses[start_ind:])), losses[start_ind:], label='Loss for PGD')
#plt.legend()

a,losses,steps_count = saga(x,y, step=0.001)
print('Number of nonzero coeff: {}'.format(np.sum(a!=0)))
print('Squared error for betas: {}'.format(np.sum((a-beta)**2)))
print('Steps count for SAGA2: {}\nThe loss was {}'.format(steps_count, losses[-1]))
plt.figure(1)
plt.plot(np.arange(len(losses[start_ind:])), losses[start_ind:], label='Loss for SAGA')

plt.legend()
plt.show()
