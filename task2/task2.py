import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

def findSteadyState(system_number, alpha):
	if system_number == 6:
		steady_states = [np.sqrt(alpha), - np.sqrt(alpha)]
	elif system_number == 7:
		steady_states = []
		if (alpha - 3) > 0:  # because unless, we will not have steady states
			steady_states = [np.sqrt((alpha-3)/2.0), - np.sqrt((alpha-3)/2.0)]

	return steady_states


def getPoints(system_number, alphas):
	# TODO: find which one is stable which is unstable
	x_values_first = []  
	x_values_second = []
	for a in alphas:
		steady_states = findSteadyState(system_number,a)
		if steady_states != []:
			x_values_first.append(steady_states[0])
			x_values_second.append(steady_states[1])

	return x_values_first, x_values_second

def drawBifurcationDiagram(system_number, alpha_min, alpha_max, alpha_step_size):
	alphas = np.arange(alpha_min, alpha_max, alpha_step_size)  # when you increase, bifurcation point does not appear on the plot
	# taking alpha values that makes steady point positive (inside square root)
	if system_number == 6:
		alpha_positives = alphas[alphas >= 0]  # probably due to numerical issues, 0 is not present in this array
	elif system_number == 7:
		alpha_positives = alphas[alphas >= 3]  

	x0_1, x0_2 = getPoints(system_number, alpha_positives) # sending only alphas that satisfy creation of steady points

	# TODO: make plots better, fix alpha range

	if x0_1 != [] and x0_2 != []:
		plt.plot(alpha_positives, x0_1, color = 'k')
		plt.plot(alpha_positives, x0_2, color = 'k')

	plt.xlabel("alpha")
	plt.ylabel("x")
	plt.title(("Bifurcation Diagram of System {}").format(system_number))
	plt.show()


def main():
	alpha_min = -4
	alpha_max = 4.004
	alpha_step_size = 0.004
	system_number = 7
	drawBifurcationDiagram(system_number, alpha_min, alpha_max, alpha_step_size) # 6 for system 6, 7 for system 7
	


if __name__ == '__main__':
  main()


'''
     **** We can use solver to show the attracted and repelled points

def system6(t,x,alpha):
	return alpha - x**2

def system7(t,x,alpha):
	return alpha - 2*(x**2) - 3



def findStability(num,alpha):
	
	steady_states = findSteadyState()
	

	print("steady_states: ", steady_states)
	real_steady_states = steady_states.real
	print("real_steady_states: ", real_steady_states)

	num_positive = 0
	num_negative = 0
	for value in real_steady_states:
		if value > 0:
			num_positive += 1
		else:
			num_negative += 1

	if num_negative == num_positive:
		return 0 # saddle point
	elif num_negative > num_positive:
		return 1 # stable point
	else:
		return -1 # unstable point

alphas = np.arange(-1,1.0001,0.0001)
empty_alphas = np.zeros(len(alphas))

for a in alphas:
	#time = (0.0, 1000.0)
	#x_init = np.arange(-1, 1.1, 0.1)
	#x_init = -1
	#x_init = [np.random.random()]

	#sol = solve_ivp(system6, time, x_init, args=(a,))
'''
