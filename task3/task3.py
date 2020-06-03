import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
from scipy.integrate import solve_ivp


def draw_phase_portrait(alpha):
    x = np.arange(-2, 2.01, 0.01)
    x1, x2 = np.meshgrid(x, x)
    y1 = alpha * x1 - x2 - x1 * (x1 ** 2 + x2 ** 2)
    y2 = x1 + alpha * x2 - x2 * (x1 ** 2 + x2 ** 2)

    fig = plt.figure(figsize=(6, 6))
    ax0 = fig.add_subplot()
    ax0.streamplot(x1, x2, y1, y2, color='r', linewidth=1)
    ax0.set_title("alpha = {}".format(alpha))

    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.show()


def calculate_next_state(t, x):
    v = [(x[0] - x[1] - x[0] * (x[0] ** 2 + x[1] ** 2)),
         (x[0] + x[1] - x[1] * (x[0] ** 2 + x[1] ** 2))]

    x_new = [t * v[0] + x[0], t * v[1] + x[1]]

    return x_new


def plot_by_euler_method(x, end_time):
    time_step = 0.001
    points = []
    time = 0
    while time < end_time:
        x = calculate_next_state(t=time_step, x=x)
        points.append(x)
        time += time_step

    return [i[0] for i in points], [i[1] for i in points]


def plot_by_svd_solver(x, end_time):
    t = 0
    sol = solve_ivp(calculate_next_state, [t, t + end_time], np.array(x))
    return sol.y


def cuspFunction(x, alpha1, alpha2):
    return alpha1 + alpha2 * x - x ** 3


def cuspBifurcation():
    fig = plt.figure(figsize=(20, 10))
    ax0 = fig.gca(projection='3d')
    alpha = np.arange(-1, 1.01, 0.01)
    x_values = np.arange(-1, 1.01, 0.01)
    alpha1, alpha2 = np.meshgrid(alpha, alpha)

    for z in x_values:
        X, Y = alpha1, alpha2
        Z = cuspFunction(z, X, Y)
        cset = ax0.contour(X, Y, Z + z, [z], zdir='z', colors="brown", antialiased=True, linestyles='dotted')

    plt.xlabel("alpha1")
    plt.ylabel("alpha2")
    plt.title("Cusp Bifurcation")
    plt.show()


def main():
    # cuspBifurcation()

    alphas = [-1, 0, 1]
    for alpha in alphas:
        draw_phase_portrait(alpha)

    fig = plt.figure(figsize=(6, 6))
    end_time = 400

    initial_points = [[2, 0], [0.5, 0]]
    for index, x in enumerate(initial_points):
        euler_sol = plot_by_euler_method(x, end_time)
        plt.scatter(euler_sol[0], euler_sol[1], linewidth=1, s=1)
        plt.scatter(euler_sol[0][0], euler_sol[1][0], color="k", s=10)
        plt.scatter(euler_sol[0][-1], euler_sol[1][-1], color="r", s=10)
        plt.title("Euler method(Initial point: {})".format(initial_points[index]))
        plt.xlabel("x1")
        plt.ylabel("x2")
        plt.xlim(-2, 2)
        plt.ylim(-2, 2)
        plt.axis('equal')
        plt.show()

        svd_sol = plot_by_svd_solver(x, end_time)
        plt.scatter(svd_sol[0], svd_sol[1], s=1)
        plt.scatter(svd_sol[0][0], svd_sol[1][0], color="k", s=10)
        plt.scatter(svd_sol[0][-1], svd_sol[1][-1], color="r", s=10)
        plt.title("IVP method(Initial point: {})".format(initial_points[index]))
        plt.xlabel("x1")
        plt.ylabel("x2")
        plt.xlim(-2, 2)
        plt.ylim(-2, 2)
        plt.axis('equal')
        plt.show()


if __name__ == '__main__':
    main()
