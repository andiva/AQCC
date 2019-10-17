import numpy as np
from scipy import sparse
import sympy
import matplotlib.pyplot as plt
import matplotlib
import progressbar
from cycler import cycler
import time
from sklearn.metrics import mean_squared_error as MSE


N = 1000
dx = 2*np.pi/N
dt = 2.5e-4
dt_map = 1.25e-3
nu=0.05

is_train_PNN = False # set to True for additional training

x = np.arange(0, 2*np.pi, dx)
t = np.arange(0, 0.5+dt, dt)
t_map = np.arange(0, 0.5+dt_map, dt_map)

#-------------------------------------------------------------------------------
def analytic_method():
    X, T = sympy.symbols('X T')
    phi = sympy.exp(-(X - 4 * T)**2 / (4 * nu * (T + 1))) + sympy.exp(-(X - 4 * T - 2 * np.pi)**2 / (4 * nu * (T + 1)))
    dphidx = phi.diff(X)

    u_analytic_ = -2 * nu / phi * dphidx + 4
    u_analytic  = sympy.utilities.lambdify((X, T), u_analytic_)

    return u_analytic


analytic_solution = analytic_method()


def u_analytic(t, x):
    ''' Analytic solution provided as benchmark in Problem Statement
    '''
    #return u_analytic_2(t, x)
    return analytic_solution(x, t)


def u_analytic_2(t, x):
    ''' Another analytic solution
    '''
    ksi = x-0.5*t
    return (1./(1+np.exp(0.5*ksi/nu)))


def FDM(ui):
    ''' FDM with first order (Euler) approximation
    '''
    u0 = ui[:-2]
    u1 = ui[1:-1]
    u2 = ui[2:]
    return u1 + dt*( -u1*(u1-u0)/dx + nu*(u2-2*u1+u0)/dx**2)


def get_PNN_weights(N, dt):
    ''' Calculates Taylor map based on Euler integration
    '''
    M = np.zeros((N*2, N*2))
    I = np.eye(N)
    M[:N, :N] = I
    M[:N, N:] = I*dt
    M[N:, N:] = I

    U = np.zeros((N, N))
    U[0, 0:3] = np.array([1., -2., 1.])/2
    U[1, 0:4] = np.array([2., -3., 0,  1.])/4
    U[-2, -4:] = np.array([1., 0, -3., 2.])/4
    U[-1, -3:] = np.array([1., -2., 1.])/2

    ar = np.array([1.0, 0, -2, 0, 1.])/4
    for i in range(2, N-2):
        U[i, i-2:i+3] = ar

    M[N:, N:] += dt*U*nu/dx**2

    return M


def calc_PNN(weights):
    M = sparse.csr_matrix(weights)
    u_num = np.empty((t_map.shape[0], x.shape[0]))
    u_num[0] = u_analytic(0, x)
    u_num[:, 0] = u_analytic(t_map, x[0])
    u_num[:, -1] = u_analytic(t_map, x[-1])

    xu = np.empty((2000))
    xu[:1000] = x
    start = time.time()
    for i in range(1, t_map.shape[0]):
        xu[1000:] = u_num[i-1]
        xun = M.dot(xu)
        un = np.interp(x, xun[:1000], xun[1000:])
        u_num[i, 1:-1] = un[1:-1]
    end = time.time()
    return u_num, end-start


def error(us):
    ''' Loss function for training PNN
    '''
    du_dt = np.gradient(us, axis=0)/dt
    du_dx = np.gradient(us, axis=1)/dx
    du2_dx2 = np.gradient(du_dx, axis=1)/dx
    er = du_dt + us*du_dx - nu*du2_dx2
    return np.sum(er**2)


def F(M0):
    ''' Utility for minimization '''
    M = get_PNN_weights(N, dt_map)
    M[M!=0] = M0
    M = sparse.csr_matrix(M)
    u_num, _ = calc_PNN(M)
    return error(u_num)


def train_PNN(M):
    M0 = M[M!=0] # this mask should be estimated via Quantum Regularization
    M0 = coordinate_descent_minimization(M0, N=1, step=1e-6)
    M[M!=0] = M0
    return M


def coordinate_descent_minimization(M0, N=1, step=1e-8):
    F0 = F(M0)
    for n in range(N):
        for i in progressbar.progressbar(range(len(M0))):
            M0[i]-=step
            F1 = F(M0)
            if F1<F0:
                F0=F1
    return M0


def set_colors(start_color="blue"):
    cvals  = [0., 1]
    colors = [start_color, "violet"]
    norm=plt.Normalize(min(cvals),max(cvals))
    tuples = list(zip(map(norm,cvals), colors))
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", tuples)

    ax = plt.axes()
    ax.set_prop_cycle(None)

    colors = [cmap(i) for i in np.linspace(0, 1, 8)[:6][::-1]]
    ax.set_prop_cycle(cycler('color', colors))
    return


def plot(x, u, dt, label, linestyle='solid', alpha=1, time_interval=0.1):
    step = int(time_interval/dt)
    u = u[::step]
    for un in u[:-1]:
        plt.plot(x, un, linestyle=linestyle, alpha=alpha)
    plt.plot(x, un, linestyle=linestyle, label=label, alpha=alpha)
    return
#-------------------------------------------------------------------------------


def main():
    u0 = u_analytic(0, x)

    u_prec = np.empty((t.shape[0], x.shape[0]))
    u_prec[0] = u_analytic(0, x)
    for i in range(1, t.shape[0]):
        u_prec[i] = u_analytic(t[i], x)

    u_fdm = np.empty((t.shape[0], x.shape[0]))
    u_fdm[0] = u_analytic(0, x)
    u_fdm[:, 0] = u_analytic(t, x[0])
    u_fdm[:, -1] = u_analytic(t, x[-1])

    start = time.time()
    for i in range(1, t.shape[0]):
        u_fdm[i, 1:-1] = FDM(u_fdm[i-1])
    end = time.time()
    print('elapsed time (FDM):', end-start)
    print('MSE (FDM):', MSE(u_fdm[-1], u_prec[-1]))


    weights = get_PNN_weights(N, dt_map)
    if is_train_PNN:
        weights = train_PNN(weights)

    u_map, elapsed_time = calc_PNN(weights)

    print('elapsed time (PNN):', elapsed_time)
    print('MSE (PNN):', MSE(u_map[-1], u_prec[-1]))


    set_colors("red")
    plot(x, u_map, dt_map, 'PNN (dt=1.25e-3)', linestyle='dashed')

    set_colors()
    plot(x, u_fdm, dt, 'FDM (dt=2.5e-4)')
    plot(x, u_prec, dt, 'Analytic', linestyle='dashed', alpha=0.5)


    plt.grid()
    plt.ylim([0, 10])
    plt.xlim([0, 2*np.pi])
    plt.xlabel('x')
    plt.ylabel('u(t,x)')
    plt.legend()
    plt.show()
    return 0


if __name__ == "__main__":
    main()


