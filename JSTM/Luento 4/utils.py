import numpy as np
from scipy.linalg import toeplitz

def convmtx(h,n):
    return toeplitz(np.hstack([h, np.zeros(n-1)]), np.hstack([h[0], np.zeros(n-1)])).T


def lmsalg(x, d, p, mu, NLMS=False, w0=None):
    q = p + 1
    if w0 is None:
        w0 = np.zeros(q)

    N = len(x)
    X = convmtx(x, q)
    e = np.zeros(N)
    y = np.zeros(N)
    What = np.zeros((N+1, q))
    What[q,:] = w0
    den = 1.0
    for ii in range(q,N):
        y[ii] = What[ii, :]@X[:, ii]
        e[ii] = d[ii] - y[ii]
        if NLMS:
            den = np.linalg.norm(X[:,ii])**2 + 1e-3
        What[ii+1, :] = What[ii, :] + mu*e[ii]*X[:,ii]/den
    y[:p] = w0[np.newaxis, :]@X[:,:p]
    e[:p] = d[:p] - y[p]
    w = What[-1,:]
    return What, y, e, w


def rms(x):
    return np.sqrt(np.mean(x**2))

def wiener(x, d, p):
    _, rx = xcorr(x, maxlags=p, scaleopt="biased")
    _, rdx = xcorr(d, x, maxlags=p, scaleopt="biased")

    Rx = toeplitz(rx[p:])
    w = np.linalg.lstsq(Rx, rdx[p:][:,np.newaxis], rcond=-1)[0].flatten()
    return w

def xcorr(x, y=None, maxlags=None, scaleopt="biased"):

    N = len(x)

    if y is None:
        y = x

    if N != len(y):
        raise ValueError('x and y must be equal length')

    c = np.correlate(x, y, mode='full')

    if maxlags is None:
        maxlags = N - 1

    if maxlags >= N or maxlags < 1:
        raise ValueError(f"maxlags must be strictly positive < {N}")

    lags = np.arange(-maxlags, maxlags + 1)

    c = c[N - 1 - maxlags:N + maxlags]

    if scaleopt == "unbiased":
        c = c/(N - np.abs(lags))

    elif scaleopt == "biased":
        c = c/N
    else:
        raise ValueError(f"Unknown type: {type}. Only 'biased' and 'unbiased' supported.")

    return lags, c
