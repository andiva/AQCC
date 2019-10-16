import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from cycler import cycler
import time

dx = 0.001
T = 0.5
dt = 5e-4


def u_0(x, b=0.5, c=0.1):
    return np.exp(-(x-b)**2/(2*c**2))


def FDM(ui):
    ''' Finite Difference Method: Lax-Friedrichs numerical scheme
    '''
    u0 = ui[:-2]
    u1 = ui[1:-1]
    u2 = ui[2:]

    return 0.5*(u2+u0) -dt*0.5*0.5*(u2*u2-u0*u0)/dx


def solution_map(x0, u0, dt):
    return x0+dt*u0


def area(x,u):
    return 0.5*np.abs(np.dot(x,np.roll(u,1))-np.dot(u,np.roll(x,1)))


def equal_area(x, u):
    ''' Simple brute-force search for discontinuty location
        todo: require code optimization
    '''
    dx = np.diff(x)
    lower_bound_x = np.argmax(dx<=0)+1
    if lower_bound_x==1:
        return np.ones_like(x, dtype=bool)

    indices = np.arange(0, len(x))
    upper_bound_x = np.argmax(np.logical_and(dx>=0, indices[1:] > lower_bound_x))+1

    min_diff_area = np.PINF
    x_discontinuity = 0

    for i in range(lower_bound_x, upper_bound_x+1, 1):
        start_i = np.argmax(x>=x[i])
        end_i = np.argmax(np.logical_and(x>=x[i], indices > i))
        right_area = np.logical_and(indices>=start_i, indices<=i)
        left_area = np.logical_and(indices>=i, indices<=end_i)

        right_area = area(x[right_area], u[right_area])
        left_area = area(x[left_area], u[left_area])

        diff_area = np.abs(right_area-left_area)

        if diff_area <= min_diff_area:
            min_diff_area = diff_area
            i_discontinuity = i

    i = i_discontinuity
    start_i = np.argmax(x>=x[i])
    end_i = np.argmax(np.logical_and(x>=x[i], indices > i))
    return np.logical_or(indices<=start_i, indices>=end_i)


def set_colors():
    ax = plt.axes()
    colors = [plt.cm.rainbow(i) for i in np.linspace(0, 1, 11)]
    ax.set_prop_cycle(cycler('color', colors))
    return


def main():
    x = np.arange(0, 1.02, dx)
    t = np.arange(0, T+dt, dt)
    u0 = u_0(x)
    u_fdm = np.empty((t.shape[0], x.shape[0]))
    u_fdm[0] = u0

    n = 10 # number of instants
    step = int(len(t)/n)

    # the proposed method converges to semi-analytical solution
    for k in range(n+1):
        xk = solution_map(x, u0, k*step*dt)
        mask = equal_area(xk, u0)
        plt.plot(xk[mask], u0[mask], 'b--', alpha=0.6)
    set_colors()
    #---------------------------------------------------------------------------

    # the FDM method introduces nonphysical dissipation
    for i in range(1, t.shape[0]):
        u_fdm[i, 1:-1] = FDM(u_fdm[i-1])

    for k in range(n+1):
        plt.plot(x, u_fdm[k*step], label="t = "+str(k*step*dt))



    plt.legend()
    plt.grid()
    plt.show()
    return 0

if __name__ == "__main__":
    main()
