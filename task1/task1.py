import numpy as np
import matplotlib.pyplot as plt
from numpy import linalg as LA


def phasePortrait(alpha):
	"""
	Draws phase portrait of the system.

	:param alpha: Alpha value to draw corresponding phase portrait.
	"""
	x = np.arange(-1, 1.1, 0.1)
	x1, x2 = np.meshgrid(x, x)

	Ax_1 = alpha * x1 + alpha * x2
	Ax_2 = (-1/4) * x1

	fig = plt.figure()
	ax0 = plt.gca()
	ax0.streamplot(x1, x2, Ax_1, Ax_2, color='dodgerblue', linewidth=1, )
	ax0.set_title("Apha = {}".format(alpha))
	ax0.set_xlim([-1,1])
	ax0.set_ylim([-1,1])
	plt.xlabel("x1")
	plt.ylabel("x2")
	plt.show()


def findEigenvalues(alpha):
	"""
	Computes eigenvalues of the given parameterized matrix.

	:param alpha: Parameter alpha of the matrix.
	:return: Eigenvalues of the matrix.
	"""
	A = np.array([[alpha,alpha], [-1/4, 0]])
	values, vectors = LA.eig(A)
	return values

def main():
	# Just to see if we have any eigenvalues which are both negative (We don't)
	'''
	alphas = np.arange(-10,10,0.00001)
	for alp in alphas:
		eigvalues = findEigenvalues(alp)
		if eigvalues[0] and eigvalues[1] < 0:
			print(eigvalues)
	'''

	alpha = -0.1
	phasePortrait(alpha)
	eigvalues = findEigenvalues(alpha)
	print(eigvalues)
	


if __name__ == '__main__':
  main()

