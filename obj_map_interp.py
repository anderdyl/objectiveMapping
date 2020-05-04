


import numpy as np
import scipy.sparse as spa
import scipy
import time

def map_interp(x, y, zFluc, noise, Lx, Ly, xInt, yInt):
    """ the main objective mapping function, converted from Bonnie Ludka's matlab script 09/2019
    Using gaussian with length scales Lx and Ly and user estimated noise_var.
    Uses fluctuation data zfluc (mean already removed by user) defined at x,y coordinates to
    interpolate to coordinates xInt, yInt. Outputs map fluctuation mapfluc and normalized mean
    square error estimate nmseEst at xInt, yInt coords

    Args:
        x (array):
        y (array):
        zFluc (array):
        noise (array):
        Lx (int):
        Ly (int):
        xInt
        yInt

    Returns:
        dgcov, dcovE, A, Aprime, mapFluc, nmseEst, dcovA, dcovA2, sigVar

          dictionary with following keys for all gauges
            'dgcov' (array):
            'dcovE' (array):
            'A' (array):
            'Aprime' (array):
            'mapFluc' (array):
            'nmseEst' (array):
            'dcovA' (array):
            'dcovA2' (array):
            'sigVar' (array):

    TODO:
    """


    begin = time.time()
    print(begin, 'beginning mapping function')

    sigVar = np.mean(np.square(zFluc)) - np.square(noise)
    # calculate data auto-covariance matrix
    X1, X2 = np.meshgrid(x, x)
    Y1, Y2 = np.meshgrid(y, y)
    dx = np.abs(X2-X1)
    dy = np.abs(Y2-Y1)
    dcov = sigVar*np.exp(-np.square(dy/Ly)-np.square(dx/Lx))

    small = np.nonzero(dcov < .000000001)
    dcov[small] = 0*np.ones((1, len(small[0])))
    dcov1 = spa.csr_matrix(dcov)

    dcovE = dcov1 + np.identity(len(dcov))*np.square(noise)
    dcovE = spa.csr_matrix(dcovE)


    # disp('calculate data grid covariance matrix')
    dgcov = np.zeros((len(x), len(xInt)))
    for num, j in enumerate(x):
        dx = np.abs(xInt-x[num])
        dy = np.abs(yInt-y[num])
        dgcov[num:] = sigVar*np.exp(-np.square(dy/Ly)-np.square(dx/Lx))


    small2 = np.nonzero(dgcov < .0000000001)
    dgcov[small2] = 0*np.ones((1, len(small2[0])))
    dgcov = spa.csr_matrix(dgcov)

    tic = time.time()
    A = scipy.linalg.lstsq(dcovE.todense(), dgcov.todense())
    toc = time.time()
    print(toc-tic, 'sec elapsed for matrix A')
    Aprime = np.transpose(A[0])
    mapFluc = np.dot(Aprime, zFluc)

    nmseEst = np.zeros((len(xInt), 1))
    dcovA = dcov1.dot(A[0])
    dcovA2 = spa.csr_matrix(dcovA)

    tic = time.time()
    for num, j in enumerate(xInt):
        nmseEst[num] = (np.transpose(A[0][:, num]).dot((dcovA2[:, num].todense() - 2 * dgcov[:, num].todense())) + sigVar)/sigVar
    toc = time.time()
    print(toc-tic, 'sec elapsed for error matrix')


    return dgcov, dcovE, A, Aprime, mapFluc, nmseEst, dcovA, dcovA2, sigVar
