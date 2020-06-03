import matplotlib.pyplot as plt
import numpy as np


def findSteadyState(system_number, alpha):
    if system_number == 6:
        steady_states = [np.sqrt(alpha), - np.sqrt(alpha)]
    elif system_number == 7:
        steady_states = []
        if (alpha - 3) > 0:  # because unless, we will not have steady states
            steady_states = [np.sqrt((alpha - 3) / 2.0), - np.sqrt((alpha - 3) / 2.0)]

    return steady_states


def getPoints(system_number, alphas):
    x_values_first = []
    x_values_second = []
    for a in alphas:
        steady_states = findSteadyState(system_number, a)
        if steady_states != []:
            x_values_first.append(steady_states[0])
            x_values_second.append(steady_states[1])

    return x_values_first, x_values_second


def drawBifurcationDiagram(system_number, alpha_min, alpha_max, alpha_step_size):
    alphas = np.linspace(alpha_min, alpha_max, alpha_step_size,
                         dtype=np.float128)  # when you increase, bifurcation point does not appear on the plot
    # taking alpha values that makes steady point positive (inside square root)
    print(alphas[1000])
    print(alphas[1001])
    print(alphas[1002])
    if system_number == 6:
        alpha_positives = alphas[alphas >= 0]  # probably due to numerical issues, 0 is not present in this array
    elif system_number == 7:
        alpha_positives = alphas[alphas >= 3]

    x0_1, x0_2 = getPoints(system_number, alpha_positives)  # sending only alphas that satisfy creation of steady points

    # TODO: make plots better, fix alpha range, show half-stable point

    if x0_1 != [] and x0_2 != []:
        plt.plot(alpha_positives, x0_1, color='k', label='Stable')  # stables
        plt.plot(alpha_positives, x0_2, color='k', linestyle=':', label='Unstable')  # unstables

    plt.xlabel("alpha")
    plt.ylabel("x")
    plt.title(("Bifurcation Diagram of System {}").format(system_number))
    plt.legend()
    plt.show()


def draw_sys_6_phase_portrait():
    fig = plt.figure(figsize=(9, 3))
    x = np.arange(-2, 2.01, 0.01)

    # Alpha = -1
    alpha = -1
    y = alpha - x ** 2
    ax = fig.add_subplot(1, 3, 1)

    ax.plot(x, y, color='r', linewidth=2)
    plt.plot([0], [0], 'go-', color="white")
    plt.title("alpha = {}".format(alpha), y=-0.1)
    plt.xlabel("x")
    plt.ylabel("y")
    ax.spines['left'].set_position('zero')
    ax.spines['right'].set_color('none')
    ax.spines['bottom'].set_position('zero')
    ax.spines['top'].set_color('none')
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')
    ax.annotate('', xy=(0.5, 0), xytext=(1.5, 0), arrowprops=dict(arrowstyle="->", connectionstyle="arc3", lw=2), )

    # Alpha = 0
    alpha = 0
    y = alpha - x ** 2
    ax = fig.add_subplot(1, 3, 2)

    ax.plot(x, y, color='r', linewidth=2)
    plt.plot([0], [0], 'go', color="black")
    plt.title("alpha = {}".format(alpha), y=-0.1)
    plt.xlabel("x")
    plt.ylabel("y")
    ax.spines['left'].set_position('zero')
    ax.spines['right'].set_color('none')
    ax.spines['bottom'].set_position('zero')
    ax.spines['top'].set_color('none')
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')
    ax.annotate('', xy=(1, 0), xytext=(2, 0), arrowprops=dict(arrowstyle="->", connectionstyle="arc3", lw=2), )
    ax.annotate('', xy=(0, 0), xytext=(1, 0), arrowprops=dict(arrowstyle="->", connectionstyle="arc3", lw=2), )
    ax.annotate('', xy=(-1, 0), xytext=(0, 0), arrowprops=dict(arrowstyle="->", connectionstyle="arc3", lw=2), )
    ax.annotate('', xy=(-2, 0), xytext=(-1, 0), arrowprops=dict(arrowstyle="->", connectionstyle="arc3", lw=2), )

    # Alpha = 1
    alpha = 1
    y = alpha - x ** 2
    ax = fig.add_subplot(1, 3, 3)

    ax.plot(x, y, color='r', linewidth=2)
    plt.plot([0], [0], 'go', color="black")
    plt.title("alpha = {}".format(alpha), y=-0.1)
    plt.xlabel("x")
    plt.ylabel("y")
    ax.spines['left'].set_position('zero')
    ax.spines['right'].set_color('none')
    ax.spines['bottom'].set_position('zero')
    ax.spines['top'].set_color('none')
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')
    ax.annotate('', xy=(1, 0), xytext=(2, 0), arrowprops=dict(arrowstyle="->", connectionstyle="arc3", lw=2), )
    ax.annotate('', xy=(1, 0), xytext=(0, 0), arrowprops=dict(arrowstyle="->", connectionstyle="arc3", lw=2), )
    ax.annotate('', xy=(-1, 0), xytext=(0, 0), arrowprops=dict(arrowstyle="->", connectionstyle="arc3", lw=2), )
    ax.annotate('', xy=(-1, 0), xytext=(-2, 0), arrowprops=dict(arrowstyle="->", connectionstyle="arc3", lw=2), )

    plt.show()

def draw_sys_7_phase_portrait():
    fig = plt.figure(figsize=(9, 3))
    x = np.arange(-10, 10.01, 0.01)

    # Alpha = 0
    alpha = 0
    y = alpha - 2 * (x ** 2) - 3
    ax = fig.add_subplot(1, 3, 1)

    ax.plot(x, y, color='r', linewidth=2)
    plt.title("alpha = {}".format(alpha), y=-0.1)
    plt.xlabel("x")
    plt.ylabel("y")
    ax.spines['left'].set_position('zero')
    ax.spines['right'].set_color('none')
    ax.spines['bottom'].set_position('zero')
    ax.spines['top'].set_color('none')
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')

    ax.annotate('', xy=(4, 0), xytext=(8, 0), arrowprops=dict(arrowstyle="->", connectionstyle="arc3", lw=2), )

    # Alpha = 3
    alpha = 3
    y = alpha - 2 * (x ** 2) - 3
    ax = fig.add_subplot(1, 3, 2)

    ax.plot(x, y, color='r', linewidth=2)
    plt.title("alpha = {}".format(alpha), y=-0.1)
    plt.xlabel("x")
    plt.ylabel("y")
    ax.spines['left'].set_position('zero')
    ax.spines['right'].set_color('none')
    ax.spines['bottom'].set_position('zero')
    ax.spines['top'].set_color('none')
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')
    plt.plot([0], [0], 'go', color="black")

    ax.annotate('', xy=(4, 0), xytext=(8, 0), arrowprops=dict(arrowstyle="->", connectionstyle="arc3", lw=2), )
    ax.annotate('', xy=(0, 0), xytext=(4, 0), arrowprops=dict(arrowstyle="->", connectionstyle="arc3", lw=2), )
    ax.annotate('', xy=(-4, 0), xytext=(0, 0), arrowprops=dict(arrowstyle="->", connectionstyle="arc3", lw=2), )
    ax.annotate('', xy=(-8, 0), xytext=(-4, 0), arrowprops=dict(arrowstyle="->", connectionstyle="arc3", lw=2), )

    # Alpha = 35
    alpha = 35
    y = alpha - 2 * (x ** 2) - 3
    ax = fig.add_subplot(1, 3, 3)

    ax.plot(x, y, color='r', linewidth=2)
    plt.title("alpha = {}".format(alpha), y=-0.1)
    plt.xlabel("x")
    plt.ylabel("y")
    ax.spines['left'].set_position('zero')
    ax.spines['right'].set_color('none')
    ax.spines['bottom'].set_position('zero')
    ax.spines['top'].set_color('none')
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')
    # Fixed points
    plt.plot([-4], [0], 'go', color="black")
    plt.plot([4], [0], 'go', color="black")

    ax.annotate('', xy=(4, 0), xytext=(0, 0), arrowprops=dict(arrowstyle="->", connectionstyle="arc3", lw=2), )
    ax.annotate('', xy=(4, 0), xytext=(8, 0), arrowprops=dict(arrowstyle="->", connectionstyle="arc3", lw=2), )
    ax.annotate('', xy=(-4, 0), xytext=(0, 0), arrowprops=dict(arrowstyle="->", connectionstyle="arc3", lw=2), )
    ax.annotate('', xy=(-4, 0), xytext=(-8, 0), arrowprops=dict(arrowstyle="->", connectionstyle="arc3", lw=2), )

    plt.show()

"""
def draw_sys_7_phase_portrait(alpha):
    x = np.arange(-2, 2.01, 0.01)
    y = alpha - 2 * (x ** 2) - 3

    fig = plt.figure()
    ax = fig.add_subplot()
    ax.plot(x, y, color='r', linewidth=2)
    plt.plot([0], [0], 'bo', color="white", lw=1, label="weakly stable")
    ax.set_title("alpha = {}".format(alpha))

    plt.xlabel("x")
    plt.ylabel("y")

    ax.spines['left'].set_position('zero')
    ax.spines['right'].set_color('none')
    ax.spines['bottom'].set_position('zero')
    ax.spines['top'].set_color('none')
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')

    ax.annotate('', xy=(0.5, 0), xytext=(1.5, 0), arrowprops=dict(arrowstyle="->", connectionstyle="arc3", lw=2), )

    plt.show()
"""

def main():
    # Usages
    """
    #
    #draw_sys_6_phase_portrait()
    """

    draw_sys_7_phase_portrait()


    """
    alpha_min = -1
    alpha_max = 1
    alpha_step_size = 2001
    system_number = 6
    drawBifurcationDiagram(system_number, alpha_min, alpha_max, alpha_step_size)  # 6 for system 6, 7 for system 7
    """


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
