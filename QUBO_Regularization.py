import numpy as np
import matplotlib.pyplot as plt
import os, time

import keras
from keras import backend as K
from keras.models import Sequential
from keras import regularizers

from Taylor_Map import TaylorMap
import Van_der_Pol as vdp


def create_PNN(custom_loss=None, inputDim=2, outputDim=2, order=3):
    ''' Creates polynomial neural network based on Taylor map'''
    model = Sequential()
    model.add(TaylorMap(output_dim = outputDim, order=order,
                      input_shape = (inputDim,)
              ))
    if custom_loss=='l1':
        rate = 0.0001
        def l1_reg(weight_matrix):
            return rate * K.sum(K.abs(weight_matrix))

        wts = model.layers[-1].trainable_weights # -1 for last dense layer.
        reg_loss = l1_reg(wts[0])
        for i in range(1, len(wts)):
            reg_loss += l1_reg(wts[i])

        def custom_loss(reg_loss):
            def orig_loss(y_true, y_pred):
                return K.mean(K.square(y_pred - y_true)) + reg_loss
            return orig_loss

        loss_function = custom_loss(reg_loss)
    else:
        loss_function='mean_squared_error'

    # opt = keras.optimizers.Adamax(lr=0.02, beta_1=0.99,
    #                               beta_2=0.99999, epsilon=1e-1, decay=0.0)
    model.compile(loss=loss_function, optimizer='adamax')
    return model


def predict(PNN, X0, epoch_n):
    ''' Predicts dynamics with PNN '''
    X = []
    X.append(X0)
    for i in range(epoch_n):
        x0 = X[-1]
        X.append(PNN.predict(X[-1]))

    return np.array(X)


def get_mask(b):
    b = b.reshape((2, -1))
    b_mask = [b[:, 0], b[:, 1:3], b[:, 3:7], b[:, 7:]]
    b_mask = [b.T for b in b_mask]
    return b_mask


def train(pnn, X, is_QUBO=False, tol=1e-10, max_epochs=100):
    loss=[]
    for i in range(max_epochs):
        history = pnn.fit(X[:-1], X[1:], epochs=1, verbose=1)
        last_loss = history.history['loss'][-1]
        loss.append(last_loss)
        if last_loss<tol:
            break
        if is_QUBO:
            W = pnn.get_weights()
            Q = vdp.calc_Q_array(X[:-1], X[1:], [w.T for w in W])
            b,_ = vdp.qbsolve(Q)
            M = get_mask(b)
            pnn.set_weights([w*m for w, m in zip(W, M)])
    return loss, i



def main():
    X0 = np.array([1.0, 4.0]) # train solution
    N = 999
    X = vdp.simulate(X0, N, rand_ampl=0)

    PNN_no_reg = create_PNN()
    PNN_QUBO_reg=create_PNN()
    PNN_l1_reg = create_PNN(custom_loss='l1')

    PNN = PNN_QUBO_reg
    loss_no_reg, N_no_reg = train(PNN_no_reg, X)
    loss_qubo_reg, N_qubo_reg = train(PNN_QUBO_reg, X, is_QUBO=True)
    loss_l1_reg, N_l1_reg = train(PNN_l1_reg, X)
    print('number of epochs without regularization: ', N_no_reg)
    print('number of epochs with QUBO regularization: ', N_qubo_reg)

    X_predict = predict(PNN, X0.reshape((1, 2)), N)[:,0,:]

    X_predict1 = predict(PNN, np.array([1.0, 4.0]).reshape((1, 2)), N)[:,0,:]
    X1 = vdp.simulate(np.array([1, 4]), N, rand_ampl=0)

    X_predict2 = predict(PNN, np.array([2.0, -2.0]).reshape((1, 2)), N)[:,0,:]
    X2 = vdp.simulate(np.array([2, -2]), N, rand_ampl=0)

    X_predict3 = predict(PNN, np.array([-3.0, 2.0]).reshape((1, 2)), N)[:,0,:]
    X3 = vdp.simulate(np.array([-3, 2]), N, rand_ampl=0)

    #print X_predict.shape

    plt.plot(X[:, 0], X[:, 1], 'b-')
    plt.plot(X1[:, 0], X1[:, 1], 'g-')
    plt.plot(X2[:, 0], X2[:, 1], 'm-')
    plt.plot(X3[:, 0], X3[:, 1], 'y-')

    plt.plot(X_predict[::10, 0], X_predict[::10, 1], 'r*')
    plt.plot(X_predict1[::10, 0], X_predict1[::10, 1], 'r*')
    plt.plot(X_predict2[::10, 0], X_predict2[::10, 1], 'r*')
    plt.plot(X_predict3[::10, 0], X_predict3[::10, 1], 'r*')

    plt.figure()
    plt.plot(loss_no_reg, label='Loss without regularization')
    plt.plot(loss_qubo_reg, label='Loss with QUBO regularization')
    plt.plot(loss_l1_reg, label='Loss with L1 regularization')
    plt.legend()
    plt.yscale("log")
    plt.grid()
    plt.show()
    return 0


if __name__ == "__main__":
    main()