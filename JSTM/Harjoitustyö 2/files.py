import numpy as np
from scipy.linalg import toeplitz, solve
from statsmodels.tsa.stattools import acovf, ccovf

def wiener(x, d, p,biased=False):
    """ Wiener filter
        
        inputs:
        -------
        x - signal to be filtered
        d - desired signal
        p - filter order
        
        outputs:
        --------
        w - filter weights
    """
    rx = acovf(x,adjusted=biased,demean=False,nlag=p)
    rdx = ccovf(d,x,adjusted=False,demean=False)
    Rx = toeplitz(rx)
    w = solve(Rx,rdx[:p+1])
    return w

def lms_update(xvec, d_n, w0, mu, NLMS=False):
    """ xvec = (x(n),...,x(n-p)), p+1 vector of past observations
        d_n = d[n], desired signal at time n
        w0 = p+1 vector of previous filter values
        mu = step size (scalar > 0)
    """
    y_n = np.dot(w0,xvec)
    e_n = d_n - y_n
    if NLMS:
        den = np.linalg.norm(xvec)**2 + 1e-3
    else:
        den = 1.0
    w1 = w0 + mu*e_n*xvec/den
    return w1,y_n,e_n

def lmsalg(x, d, p, mu, NLMS=False,w0=None):
    """ LMS algorithm in the batch mode. Returns What, y, e, w 
        
        inputs:
        -------
        x - input signal to filter
        d - desired signal
        p - filter order
        mu - step size
        NLMS - if true, use normalized LMS
        w0 - initial filter coefficients (optional)
        
        outputs:
        --------
        What - all estimated filter weights
        y - output of filter
        e - error signal
        w - current filter weights
    """
    q = p+1
    if w0 is None:
        w0 = np.zeros(q)
    N = len(x)
    e = np.zeros(N)
    y = np.zeros(N)
    What = np.zeros((N,q))
    What[p-1,:] = w0
    for ii in range(p,N):
        What[ii,:],y[ii],e[ii] =lms_update(np.flip(x[ii-p:ii+1]),d[ii],What[ii-1],mu,NLMS)
    w = What[-1,:]
    return What, y, e, w
