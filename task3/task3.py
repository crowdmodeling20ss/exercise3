import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
from scipy.integrate import solve_ivp
from mpl_toolkits.mplot3d import Axes3D 

def draw_phase_portrait(alpha):
    """
    Draws phase portrait of the system.

    :param alpha: Alpha value to draw corresponding phase portrait.
    """
    x = np.arange(-2, 2.01, 0.01)
    x1, x2 = np.meshgrid(x, x)
    y1 = alpha * x1 - x2 - x1 * (x1 ** 2 + x2 ** 2)
    y2 = x1 + alpha * x2 - x2 * (x1 ** 2 + x2 ** 2)

    fig = plt.figure()
    ax0 = fig.add_subplot()
    ax0.streamplot(x1, x2, y1, y2, color='dodgerblue', linewidth=1)
    ax0.set_title("alpha = {}".format(alpha))
    ax0.set_xlim([-2,2])
    ax0.set_ylim([-2,2])
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.show()


def calculate_next_state(t, x):
    """
    Computes next state of given function based on the previous x value.

    :param t: One-dimensional independent variable (time)
    :param x: State of the function.
    :return: New state of the function.
    """
    v = [(x[0] - x[1] - x[0] * (x[0] ** 2 + x[1] ** 2)),
         (x[0] + x[1] - x[1] * (x[0] ** 2 + x[1] ** 2))]

    x_new = [t * v[0] + x[0], t * v[1] + x[1]]

    return x_new


def plot_by_euler_method(x, end_time):
    """
    Plots trajectory using Euler Method.

    :param x: State of the function.
    :param end_time: Upper bound of simulation time.
    :return: Coordinates of the calculated points.
    """
    time_step = 0.001
    points = []
    time = 0
    while time < end_time:
        x = calculate_next_state(t=time_step, x=x)
        points.append(x)
        time += time_step

    return [i[0] for i in points], [i[1] for i in points]


def plot_by_svd_solver(x, end_time):
    """
    Plots trajectory using solve_ivp.

    :param x: State of the function.
    :param end_time: Upper bound of simulation time.
    :return: Coordinates of the calculated points.
    """
    t = 0
    sol = solve_ivp(calculate_next_state, [t, t + end_time], np.array(x))
    return sol.y


def cuspFunction(x,alpha1,alpha2):
    """
    Computes result of cusp function given its parameters.

    :param x: State of the function.
    :param alpha1: First alpha parameter of the function.
    :param alpha2: Second alpha parameter of the function.
    :return: Returns solution of the function.
    """
    return alpha1 + alpha2*x - x**3


def cuspBifurcation():
    """
    Plots cusp bifurcation.
    """
    fig = plt.figure(figsize=(10,10))
    ax0 = fig.gca(projection='3d')
    alpha = np.arange(-1, 1.01, 0.01)
    x_values = np.arange(-1, 1.01, 0.01)
    alpha1, alpha2 = np.meshgrid(alpha, alpha)

    for z in x_values:
        X,Y = alpha1,alpha2
        Z = cuspFunction(z,X,Y)
        cset = ax0.contour(X, Y, Z+z, [z], zdir='z', colors = "brown", antialiased=True, linestyles='dotted')

    ax0.set_xlabel('alpha1')
    ax0.set_ylabel('alpha2')
    ax0.set_zlabel('x')

    plt.title("Cusp Bifurcation")
    plt.show()


def main():

    cuspBifurcation()
    
    alphas = [-1,0,1]
    for alpha in alphas:
        draw_phase_portrait(alpha)

    fig = plt.figure(figsize=(8,5))
    end_time = 400
    cmap = ["BuPu", "Purples", "bwr"][1]
    initial_points = [[2, 0], [0.5, 0]]
    for index, x in enumerate(initial_points):
        euler_sol = plot_by_euler_method(x, end_time)
        plt.scatter(euler_sol[0], euler_sol[1],linewidth=1,s=1,color="teal")
        plt.scatter(euler_sol[0][0], euler_sol[1][0],color ="lightcoral",s=10, label="Starting Point")
        plt.scatter(euler_sol[0][-1], euler_sol[1][-1],color ="firebrick",s=10,label="Ending Point")
        plt.title("Euler's Method: {}".format(initial_points[index]))
        plt.xlabel("x1")
        plt.ylabel("x2")
        plt.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')
        plt.tight_layout()
        plt.show()

        svd_sol = plot_by_svd_solver(x, end_time)
        plt.scatter(svd_sol[0], svd_sol[1],s=1,color="teal")
        plt.scatter(svd_sol[0][0], svd_sol[1][0],color ="lightcoral",s=10, label="Starting Point")
        plt.scatter(svd_sol[0][-1], svd_sol[1][-1],color ="firebrick",s=10,label="Ending Point")
        plt.title("SVD Method: {}".format(initial_points[index]))
        plt.xlabel("x1")
        plt.ylabel("x2")
        plt.legend()
        plt.tight_layout()
        plt.show()


if __name__ == '__main__':
    main()
