


import numpy as np
import scipy.sparse as spa
import scipy
import time

def map_interp(x, y, zFluc, noise, Lx, Ly, xInt, yInt):
    """The main objective mapping function.
    
    Using gaussian with length scales Lx and Ly and user estimated noise_var.
    Uses fluctuation data zfluc (mean already removed by user) defined at x,y coordinates to
    interpolate to coordinates xInt, yInt. Outputs map fluctuation mapfluc and normalized mean
    square error estimate nmseEst at xInt, yInt coords

    Args:
        x (array):  describes coordinates of zFluc
        y (array):  describes coordinates of zFluc
        zFluc (array): perturbation from background mean values (mean removed already)
        noise (array):  user estimated noise
        Lx (int): gaussian length scale in x
        Ly (int): gaussian length scale in y
        xInt: interpolated to these coordinates (x)
        yInt: interpolated to these coordinates (y)

    Returns:

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
            
    Notes:
         converted from Bonnie Ludka's matlab script 09/2019
         
    
    """
    start = time.time()
    print('   beginning mapping function')
    # calculate signal variance
    sigVar = np.mean(np.square(zFluc)) - np.square(noise)
    
    # calculate data auto-covariance matrix
    X1, X2 = np.meshgrid(x, x)
    Y1, Y2 = np.meshgrid(y, y)
    dx = np.abs(X2-X1)
    dy = np.abs(Y2-Y1)
    dcov = sigVar * np.exp(-np.square(dy/Ly)-np.square(dx/Lx))

    small = np.nonzero(dcov < .000000001)
    dcov[small] = 0*np.ones((1, len(small[0])))
    dcov1 = spa.csr_matrix(dcov)

    dcovE = dcov1 + np.identity(len(dcov))*np.square(noise)
    dcovE = spa.csr_matrix(dcovE)

    # calculate data grid covariance matrix
    dgcov = np.zeros((len(x), len(xInt)))
    for num, j in enumerate(x):
        dx = np.abs(xInt-x[num])
        dy = np.abs(yInt-y[num])
        dgcov[num:] = sigVar * np.exp(-np.square(dy/Ly) - np.square(dx/Lx))

    small2 = np.nonzero(dgcov < .0000000001)
    dgcov[small2] = 0*np.ones((1, len(small2[0])))
    dgcov = spa.csr_matrix(dgcov)

    tic = time.time()
    A = scipy.linalg.lstsq(dcovE.todense(), dgcov.todense())
    print('  {:.1f} sec elapsed for matrix A'.format(time.time()-tic))
    Aprime = np.transpose(A[0])
    mapFluc = np.dot(Aprime, zFluc)

    nmseEst = np.zeros((len(xInt), 1))
    dcovA = dcov1.dot(A[0])
    dcovA2 = spa.csr_matrix(dcovA)

    tic = time.time()
    for num, j in enumerate(xInt):
        nmseEst[num] = (np.transpose(A[0][:, num]).dot((dcovA2[:, num].todense() - 2 * dgcov[:, num].todense())) + sigVar)/sigVar
    print('  {:.1f} sec elapsed for error matrix'.format(time.time()-tic))
    print('  {:.1f} total time for objective mapping'.format(time.time() - start))

    return dgcov, dcovE, A, Aprime, mapFluc, nmseEst, dcovA, dcovA2, sigVar
