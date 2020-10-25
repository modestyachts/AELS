# This is a python implementation of calfun.m, provided at http://www.mcs.anl.gov/~more/dfo/
import numpy as np
from dfovec import dfovec

def norm(x, type=2):
    if type==1:
        return np.sum(np.abs(x))
    elif type==2:
        return np.sqrt(x**2)
    else: # type==np.inf:
        return max(np.abs(x))

def calfun(x, m, nprob, probtype='smooth'):
    n = len(x)

    # Restrict domain for some nondiff problems
    xc = x
    if probtype is 'nondiff':
        if nprob==8 or nprob==9 or nprob==13 or nprob==16 or nprob==17 or nprob==18:
            xc = max(x,0)

    # Generate the vector
    fvec = dfovec(m, n, xc, nprob)

    # Calculate the function value
    if probtype is 'noisy3':
        sigma = 1e-3
        u = sigma*(-np.ones(m) + 2*np.random.rand(m))
        fvec = fvec * (1+u)
        y = np.sum(fvec**2)
    elif probtype is 'wild3':
        sigma = 1e-3
        phi = 0.9*np.sin(100*norm(x,1))*np.cos(100*norm(x,np.inf)) + 0.1*np.cos(norm(x,2))
        phi = phi*(4*phi**2 - 3)
        y = (1 + sigma*phi)*sum(fvec**2)
    elif probtype is 'smooth':
        y = np.sum(fvec**2)
    elif probtype is 'nondiff':
        y = np.sum(np.abs(fvec))
    else:
        print('invalid probtype')
        return None
    # Never return nan. Return inf instead so that optimization algorithms treat it as out of bounds.
    if np.isnan(y):
        return np.inf
    return y
