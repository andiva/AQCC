import numpy as np
from dwave_qbsolv import QBSolv


# Numerically estimated Taylor map from Van der Pol equation
W_pole = [np.array([[0.], [0.]]),
          np.array([[ 0.99995067,  0.01004917],
                    [-0.01004917,  1.00999984]]),
          np.array([[0., 0., 0., 0.],
                    [0., 0., 0., 0.]]),
          np.array([[ 1.59504733e-07, -4.94822066e-05, 0., -3.20576750e-07, 0.,0.,0., -7.90629025e-10],
                    [ 4.94821629e-05, -1.00975145e-02, 0., -9.96173322e-05, 0.,0.,0., -3.30168067e-07]])]


def qbsolve(Q, size=30):
    q_min = 0
    a_min = []
    response = QBSolv().sample_qubo(Q)
    for res in response.samples():
        a = np.empty(size)
        for key in res:
            a[key] = res[key]
        #------------------------------------------
        q = 0
        for i in xrange(size):
            for j in xrange(size):
                if (i,j) in Q:
                    q+=Q[(i,j)]*a[i]*a[j]
        if q<q_min:
            q_min=q
            a_min = a

    return a_min, q_min


def get_pows(X0):
    x = X0[0]
    y = X0[1]

    return [np.array([[1.]]),
            np.array([x, y]),
            np.array([x**2, x*y, y*x, y**2]),
            np.array([x**3, x**2*y, y*x**2, x*y**2, x**2*y, x*y**2, y**2*x, y**3])]


def mult(W, X0):
    X = get_pows(X0)

    m = np.empty((X0.shape[0], 0))
    for wi, xi in zip(W, X):
        m = np.hstack((m, wi*xi))

    return m


def simulate(X0, N, rand_ampl = 0):
    X = [X0]
    for i in xrange(N):
        X_i = mult(W_pole, X[-1]).sum(axis=1)
        r = np.random.rand(*X_i.shape)*2-1
        X.append(X_i+r*rand_ampl)
    return np.array(X)


def get_true_mask():
    a_true = np.empty((2, 0), dtype=int)
    for w in W_pole:
        a_true = np.hstack((a_true, w!=0))
    return a


def replace_zeros_by_random():
    W = []
    for w in W_pole:
        w = w.copy()
        r = np.random.rand(*w.shape)
        w[w==0] = r[w==0]
        W.append(w)
    return W


def calc_Q(X0, X1, W):
    X_pred = mult(W, X0).ravel()

    a0 = np.repeat(X1, int(X_pred.shape[0]/2))

    Q = {}
    for i in xrange(X_pred.shape[0]):
        Q[(i, i)] = X_pred[i]**2 - 2*X_pred[i]*a0[i]

        end_j = X_pred.shape[0]
        if i < int(X_pred.shape[0]/2):
            end_j = int(X_pred.shape[0]/2)

        for j in range(i+1, end_j):
            Q[(i,j)] = 2*X_pred[i]*X_pred[j]
    return Q


def sum_Q(Q0, Q1):
    Q = Q0.copy()
    for key in Q1:
        if key not in Q:
            Q[key] = 0
        Q[key]+=Q1[key]
    return Q


def calc_Q_array(X0, X1, W):
    Q = {}
    for x0, x1 in zip(X0, X1):
        Q = sum_Q(Q, calc_Q(x0, x1, W))
    return Q


def main():
    X0 = np.array([1, 2])
    X = mult(W_pole, X0).sum(axis=1)
    print('initial state: ', X0)
    print('next state:', X)

    W = replace_zeros_by_random()
    Q = calc_Q(X0, X, W)

    a, q = qbsolve(Q)
    print(a.reshape((2, -1)))
    a *= mult(W, X0).ravel()
    a = a.reshape((2, -1)).sum(axis=1)
    print(q, ':',  a)

    return 0

if __name__ == "__main__":
    main()