import numpy as np
import matplotlib.pyplot as plt
from numpy import linalg as LA

def phasePortrait(alpha):
	x = np.arange(-1, 1.1, 0.1)
	x1, x2 = np.meshgrid(x, x)
	Ax_1 = alpha * x1 + alpha * x2
	Ax_2 = (-1/4) * x1

	fig = plt.figure()
	ax0 = fig.add_subplot()
	ax0.streamplot(x1, x2, Ax_1, Ax_2, color='r', linewidth=2)
	ax0.set_title("Apha = {}".format(alpha))
	plt.show()

def findEigenvalues(alpha):
	A = np.array([[alpha,alpha], [-1/4, 0]])
	values, vectors = LA.eig(A)
	return values

def main():
	alpha = -0.3
	phasePortrait(alpha)
	eigvalues = findEigenvalues(alpha)
	print(eigvalues)
	


if __name__ == '__main__':
  main()


'''
	n+ = number of eigenvalues with positive real part
	n- = number of eigenvalues with negative real part
	n0 = number of eigenvalues with zero real part

	node = real simple eigenvalues 
	focus: complex eigenvalues

	1- alpha = 1.5 ==> [1.1830127 0.3169873] ==> two positive real eigenvalues = node, unstable. n+ = 2, n- = 0, n0 = 0
	2- alpha = 0.5 ==> [0.25+0.25j 0.25-0.25j] ==> two COMPLEX eigenvalues (with positive real part) = focus, unstable, n+ = 2, n- = 0, n0 = 0
	3- alpha = -0.3 ==> [-0.4622499  0.1622499] ==> one positive one negative real eigenvalue = saddle point, unstable n+ = 1, n- = 1, n0 = 0
	
	- Theorem 2.2, Page 48:
		The phase portraits of system (2.11) near two hyperbolic equilibria, x0 and y0, are locally topologically equivalent 
		if and only if these equilibria have the same number n− and n+ of eigenvalues with Re λ < 0 and with Re λ > 0, respectively. 

	- Nodes and foci(of corresponding stability) are topologically equivalent but can be identified looking at the eigenvalues.

	Therefore; 1 and 2 are topologically equivalent since they have the same stability and hence same number n− and n+ of eigenvalues which is n+ in this case.
'''