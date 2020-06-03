from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
from scipy.integrate import solve_ivp


def lorenzEquations(t, x0, sigma, rho, beta):
    """
    Computes result of Lorenz Equation given its parameters.

    :param t: One-dimensional independent variable (time)
    :param x0: Starting points of the system.
    :param sigma: Parameter sigma of the function.
    :param rho: Parameter rho of the function.
    :param beta: Parameter beta of the function.
    :return: Calculated value of the function.
    """
    x, y, z = x0
    dxdt = sigma * (y - x)
    dydt = x * (rho - z) - y
    dzdt = x * y - beta * z

    return dxdt, dydt, dzdt


def plotLorenz(x0_x, rho):
    """
    Plots Lorenz Equation.

    :param x0_x: x coordinate of the initial point.
    :param rho: Parameter rho of the function.
    :return: Solution of the system.
    """
    time = (0.0, 1000.0)
    x0 = [x0_x, 10, 10]
    sigma = 10
    beta = 8.0 / 3
    width = 0.1
    c = 'red'
    if x0_x > 10:
        c = 'blue'
    if rho < 28:
        width = 2

    solution = solve_ivp(lorenzEquations, time, x0, args=(sigma, rho, beta,))
    sol = solution.y

    fig = plt.figure()
    ax0 = fig.gca(projection='3d')
    ax0.plot(sol[0, :], sol[1, :], sol[2, :], linewidth=width, color=c, linestyle=':', antialiased=True)
    ax0.set_xlabel('x')
    ax0.set_ylabel('y')
    ax0.set_zlabel('z')
    ax0.set_title("Lorenz Attractor, x0 = ({},{},{})".format(x0[0], x0[1], x0[2]))
    plt.show()

    return solution


def plotLorenzWithPoints(s1y, s2y, indexes):
    """
    Plots two Lorenz Equation with different and mutual points indicated.

    :param s1y: Solution coordinates of the first system.
    :param s2y: Solution coordinates of the second system.
    :param indexes: Indexes of the points where difference occurs.
    """
    sol1_different, sol2_different, sol_same = calculatePoints(s1y, s2y, indexes)

    # Different points plot
    if len(sol1_different) or len(sol2_different) != 0:
        fig = plt.figure(figsize=(12, 6))
        ax0 = fig.gca(projection='3d')
        ax1 = fig.gca(projection='3d')
        ax3 = fig.gca(projection='3d')
        ax4 = fig.gca(projection='3d')

        ax0.plot(s1y[0, :], s1y[1, :], s1y[2, :], linewidth=0.1, color='red', linestyle=':', antialiased=True,
                 label='System 1, x0 = (10,10,10)')
        ax1.plot(s2y[0, :], s2y[1, :], s2y[2, :], linewidth=0.1, color='blue', linestyle=':', antialiased=True,
                 label='System 2, x0 = (10.00000001,10,10)')
        ax3.scatter(sol1_different[:, 0], sol1_different[:, 1], sol1_different[:, 2], c='#530127', s=0.1,
                    label='Different points of System 1')
        ax4.scatter(sol2_different[:, 0], sol2_different[:, 1], sol2_different[:, 2], c='#071c56', s=0.1,
                    label='Different points of System 2')
        ax4.set_xlabel('x')
        ax4.set_ylabel('y')
        ax4.set_zlabel('z')
        leg = plt.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')
        for legobj in leg.legendHandles:
            legobj.set_linewidth(2.0)
        plt.title("Lorenz Attractors with Different Points")
        plt.tight_layout()
        plt.show()

    # Mutual points plot
    if len(sol_same) != 0:
        print("len of same ", len(sol_same))
        fig = plt.figure(figsize=(12, 6))
        ax0 = fig.gca(projection='3d')
        ax1 = fig.gca(projection='3d')
        ax2 = fig.gca(projection='3d')

        ax0.plot(s1y[0, :], s1y[1, :], s1y[2, :], linewidth=0.1, color='red', linestyle=':', antialiased=True,
                 label='System 1, x0 = (10,10,10)')
        ax1.plot(s2y[0, :], s2y[1, :], s2y[2, :], linewidth=0.11, color='blue', linestyle=':', antialiased=True,
                 label='System 2, x0 = (10.00000001,10,10)')
        ax2.scatter(sol_same[:, 0], sol_same[:, 1], sol_same[:, 2], c="black", s=1, label='Mutual points of the systems')
        ax2.set_xlabel('x')
        ax2.set_ylabel('y')
        ax2.set_zlabel('z')
        leg = plt.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')
        for legobj in leg.legendHandles:
            legobj.set_linewidth(2.0)
        plt.title("Lorenz Attractors with Mutual Points")
        plt.tight_layout()
        plt.show()


def calculatePoints(s1y, s2y, indexes):
    """
    Calculates mutual and different points given two Lorenz Systems.

    :param s1y: Solution coordinates of the first system.
    :param s2y: Solution coordinates of the second system.
    :param indexes: Indexes of the points where difference occurs.
    :return: Different points of both systems and mutual points.
    """
    # for different points
    sol1_different = []
    sol2_different = []
    for i in indexes:
        point1 = [s1y[0][i], s1y[1][i], s1y[2][i]]
        point2 = [s2y[0][i], s2y[1][i], s2y[2][i]]
        sol1_different.append(point1)
        sol2_different.append(point2)

    # for the same points
    sol_same = []
    len_points = s1y.shape[1]
    if s1y.shape[1] > s2y.shape[1]:
        len_points = s2y.shape[1]

    for i in range(len_points):
        if i not in indexes:
            point = [s1y[0][i], s1y[1][i], s1y[2][i]]
            sol_same.append(point)

    sol1_different = np.array(sol1_different)
    sol2_different = np.array(sol2_different)
    sol_same = np.array(sol_same)

    return sol1_different, sol2_different, sol_same


def computeDifference(s1, s2):
    """
    Computes if the difference between points is larger than1. Length of the smaller array is considered.

    :param s1: Solution of the first system.
    :param s2: Solution of the second system.
    :return: Times and indexes where points are different.
    """
    s1_solution = s1.y
    s1_time = s1.t
    s2_solution = s2.y
    s2_time = s2.t

    distArr = []
    distArrIndex = []
    s1_shape = s1_solution.shape
    s2_shape = s2_solution.shape

    if (s1_shape[1] > s2_shape[1]):
        for i in range(s2_shape[1]):
            point1 = np.array([s1_solution[0][i], s1_solution[1][i], s1_solution[2][i]])
            point2 = np.array([s2_solution[0][i], s2_solution[1][i], s2_solution[2][i]])
            if 1 < np.linalg.norm(point2 - point1):  # If difference of the points is larger than one.
                distArr.append([s1_time, s2_time])
                distArrIndex.append(i)

    return distArr, distArrIndex


def logisticEquation(x, r):
    """
    Computes result of Logistic Map given its parameters.

    :param x: State of the function.
    :param r: Parameter r of the function.
    :return: New state of the function.
    """
    return r * x * (1 - x)


def plotLogistic(r_min, r_max):
    """
    Plots Logistic Map.

    :param r_min: Minimum r value to solve the system.
    :param r_max: Maximum r value to solve the system.
    """
    R = np.arange(r_min + 0.001, r_max + 0.001, 0.001)
    values = []
    r_val = []
    for r in R:
        x = np.random.random()
        for n in range(1000):  # we are saving the solution after 1000 iteration
            x = logisticEquation(x, r)
            if n > 939:  # saving last 60
                values.append(x)
                r_val.append(r)
    values = np.array(values)
    r_val = np.array(r_val)

    fig = plt.figure(figsize=(12, 7))
    ax0 = fig.add_subplot()
    ax0.scatter(r_val, values, s=0.1, color='seagreen')

    if r_min == 0.0:
        plt.axvline(x=1.0, linewidth=0.5, label="Fixed Point", c="lightpink")
    if r_max == 4.0:
        plt.axvline(x=3.0, linewidth=0.5, label="2-cycle, Pitchfork Bifurcation",
                    c="darkorange")  # 2-cycle = period doubling. first bifurcation (pitchfork)
        plt.axvline(x=3.828427, linewidth=0.5, label="3-cycle, Saddle Bifurcation",
                    c="indianred")  # 3-cycle, saddle bifurcation
        plt.axvline(x=3.449489, linewidth=0.5, label="4-cycle", c="deepskyblue")  # 4-cycle
        plt.axvline(x=3.73817, linewidth=0.5, label="5-cycle", c="slateblue")  # 5-cycle
        plt.axvline(x=3.62655316, linewidth=0.5, label="6-cycle", c="maroon")  # 6-cycle
        plt.axvline(x=3.70164076, linewidth=0.5, label="7-cycle", c="mediumvioletred")  # 7-cycle
        plt.axvline(x=3.54409, linewidth=0.5, label="8-cycle", c="slategrey")  # 8-cycle
        plt.axvline(x=3.56440726, linewidth=0.5, label="16-cycle", c="goldenrod")  # 16-cycle

    plt.legend(loc='upper left', prop={'size': 8})
    plt.xlabel("r")
    plt.ylabel("x")
    plt.title(("Bifurcation Diagram of Logistic Map, r = ({},{}]").format(int(r_min), int(r_max)))
    plt.show()


def main():
    # Lorenz Attractor

    x0_x = 10
    rho = 28
    s1 = plotLorenz(x0_x, rho)
    s2 = plotLorenz((10 + 10 ** (-8)), rho)

    differentPointArray, indexes = computeDifference(s1, s2)
    plotLorenzWithPoints(s1.y, s2.y, indexes)

    # Logistic Map

    plotLogistic(0.0, 4.0)  # (0,4]
    plotLogistic(0.0, 2.0)  # (0,2]
    plotLogistic(2.0, 4.0)  # (2,4]


if __name__ == '__main__':
    main()
