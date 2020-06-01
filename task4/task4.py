from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
from scipy.integrate import solve_ivp

def lorenzEquations(t,x0,sigma,ro,beta):
	x, y, z = x0
	dxdt = sigma * (y - x)
	dydt = x * (ro - z) - y
	dzdt = x * y - beta * z

	return dxdt, dydt, dzdt

def plotLorenz(x0_x,ro):
	time = (0.0, 1000.0)
	x0 = [x0_x,10,10]
	sigma = 10
	beta = 8.0/3
	width = 0.1
	c = 'red'
	if x0_x > 10:
		c = 'blue'
	if ro < 28:
		width = 2

	solution = solve_ivp(lorenzEquations, time, x0, args=(sigma,ro,beta))
	sol = solution.y
	print(sol)
	fig = plt.figure()
	ax0 = fig.gca(projection = '3d')
	ax0.plot(sol[0,:], sol[1,:], sol[2,:], linewidth = width, color= c, linestyle=':', antialiased=True)
	ax0.set_title("Lorenz Attractor, x0 = ({},{},{})".format(x0[0],x0[1],x0[2]))
	plt.show()

	return solution

def plotLorenzWithPoints(s1y,s2y,indexes):
	sol1_different, sol2_different, sol_same = calculatePoints(s1y,s2y, indexes)

	# Different points plot
	if len(sol1_different) or len(sol2_different) != 0:
		fig = plt.figure(figsize = (12,6))
		ax0 = fig.gca(projection = '3d')
		ax1 = fig.gca(projection = '3d')
		ax3 = fig.gca(projection = '3d')
		ax4 = fig.gca(projection = '3d')

		ax0.plot(s1y[0,:], s1y[1,:], s1y[2,:], linewidth = 0.1, color= 'red', linestyle=':', antialiased=True,label='System 1, x0 = (10,10,10)')
		ax1.plot(s2y[0,:], s2y[1,:], s2y[2,:], linewidth = 0.1, color= 'blue', linestyle=':', antialiased=True,label='System 2, x0 = (10.00000001,10,10)')
		ax3.scatter(sol1_different[:,0], sol1_different[:,1], sol1_different[:,2], c = '#530127',s=0.1, label='Different points of System 1')
		ax4.scatter(sol2_different[:,0], sol2_different[:,1], sol2_different[:,2], c = '#071c56',s=0.1, label='Different points of System 2')
		leg = plt.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')
		for legobj in leg.legendHandles:
			legobj.set_linewidth(2.0)
		plt.title("Lorenz Attractors with Different Points")
		plt.tight_layout()
		plt.show()


	# Mutual points plot
	if len(sol_same) != 0:
		fig = plt.figure(figsize = (12,6))
		ax0 = fig.gca(projection = '3d')
		ax1 = fig.gca(projection = '3d')
		ax2 = fig.gca(projection = '3d')

		ax0.plot(s1y[0,:], s1y[1,:], s1y[2,:], linewidth = 1, color= 'red', linestyle=':', antialiased=True,label='System 1, x0 = (10,10,10)')
		ax1.plot(s2y[0,:], s2y[1,:], s2y[2,:], linewidth = 1, color= 'blue', linestyle=':', antialiased=True,label='System 2, x0 = (10.00000001,10,10)')
		ax2.scatter(sol_same[:,0], sol_same[:,1], sol_same[:,2], c = "black",s=5, label='Mutual points of the systems')
		leg = plt.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')
		for legobj in leg.legendHandles:
			legobj.set_linewidth(2.0)
		plt.title("Lorenz Attractors with Mutual Points")
		plt.tight_layout()
		plt.show()


def calculatePoints(s1y,s2y,indexes):
	# for different points
	sol1_different =  []
	sol2_different = []
	for i in indexes:
		point1 = [s1y[0][i],s1y[1][i],s1y[2][i]]
		point2 = [s2y[0][i],s2y[1][i],s2y[2][i]]
		sol1_different.append(point1)
		sol2_different.append(point2)

	# for the same points
	sol_same =  []
	len_points = s1y.shape[1]
	if s1y.shape[1] > s2y.shape[1]:
		len_points = s2y.shape[1]
	
	for i in range(len_points):
		if i not in indexes:
			point = [s1y[0][i],s1y[1][i],s1y[2][i]]
			sol_same.append(point)

	sol1_different = np.array(sol1_different)
	sol2_different = np.array(sol2_different)
	sol_same = np.array(sol_same)
	return sol1_different, sol2_different, sol_same

def computeDifference(s1,s2):
	s1_solution = s1.y
	s1_time = s1.t
	s2_solution = s2.y
	s2_time = s2.t
	
	distArr = []
	distArrIndex = []
	s1_shape = s1_solution.shape
	s2_shape = s2_solution.shape
	print(s1_shape)
	print(s2_shape)
	if(s1_shape[1] > s2_shape[1]):
		for i in range(s2_shape[1]):
			point1 = np.array([s1_solution[0][i],s1_solution[1][i],s1_solution[2][i]])
			point2 = np.array([s2_solution[0][i],s2_solution[1][i],s2_solution[2][i]])
			if 1 < np.linalg.norm(point2 - point1):
				distArr.append([s1_time, s2_time])
				distArrIndex.append(i)

	return distArr, distArrIndex


def main():
	x0_x = 10
	ro = 0.5
	s1 = plotLorenz(x0_x,ro)
	s2 = plotLorenz((10+10**(-8)), ro)
	#print("s1 time: ", s1.t[-2])
	#print("s1 time: ", s2.t[-2])
	
	differentPointArray, indexes = computeDifference(s1,s2)

	print(len(differentPointArray))
	plotLorenzWithPoints(s1.y,s2.y,indexes)




if __name__ == '__main__':
	main()