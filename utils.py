import numpy as np
import time
import matplotlib.pyplot as plt
from enum import Enum
from calfun import calfun
from dfoxs import dfoxs
import os
import scipy
from scipy import optimize
import warnings

###############################################################################
# Set up the objective functions
###############################################################################

class Objective(Enum):
    LOGISTIC = 'logistic'
    DFO = 'dfo'

    def __str__(self):
        return self.value

adult_data = np.loadtxt('data/adult.csv', delimiter=',')
quantum_data = np.loadtxt('data/phy.csv', delimiter=',')
protein_data = np.loadtxt('data/bio.csv', delimiter=',')

def get_objective(objective, n=None, m=None, nprob=None, factor=None):
    if objective is Objective.LOGISTIC:
        assert(nprob is not None)
        if nprob == 0:   # Adult dataset
            data = adult_data
        elif nprob == 1: # Quantum dataset
            data = quantum_data
        elif nprob == 2: # Protein dataset
            data = protein_data
        else:
            print('Unrecognized logistic objective')
            return
        X = data[:,1:] # First entry is label
        y = data[:,0]
        n = len(X)
        x_0 = np.zeros(X.shape[1])
        lam = 1/n
        def f(x, batch=None):
            total = 0
            if batch is None:
                batch = np.arange(n)
            for i in batch:
                total = total + np.logaddexp(0.0, -y[i] * np.dot(x, X[i]))
            return lam * np.dot(x, x) / 2 + total / len(batch)
        def grad_f(x, batch=None, sigma=None, random=False):
            grad = 0
            if batch is None:
                batch = np.arange(n)
            for i in batch:
                warnings.filterwarnings("error")
                try: 
                    expon = np.exp(-y[i] * np.dot(x, X[i]))
                    grad = grad - y[i] * X[i] * expon / (1 + expon)
                except RuntimeWarning:
                    grad = grad - y[i] * X[i] # If expon overflows, approximate expon / (1 + expon) as 1
                warnings.resetwarnings()
            grad = lam * x + grad / len(batch)
            return grad
        return f, grad_f, x_0
    elif objective is Objective.DFO:
        assert(n is not None)
        assert(m is not None)
        assert(nprob is not None)
        assert(factor is not None)
        def f(x, batch=None):
            return calfun(x, m, nprob)
        def grad_f(x, batch=None, sigma=None, random=False): 
            curval = calfun(x, m, nprob)
            if random:
                u = np.random.randn(len(x))
                u = u / np.sqrt(np.dot(u, u))
                grad = (calfun(x + u * sigma, m, nprob) - curval) * u / sigma
                return grad
            grad = np.zeros_like(x)
            for i in range(len(grad)):
                e_i = np.array(np.zeros_like(x))
                e_i[i] = sigma
                grad[i] = (calfun(x + e_i, m, nprob) - curval) / sigma # Forward finite differencing
            return grad
        x0 = dfoxs(n, nprob, factor)
        return f, grad_f, x0
    else:
        print('Unrecognized objective')

class F:
    def __init__(self, objective, name=None, epsilon=0.0001, batch_size=None, T_0=1, m=None, n=None, nprob=None, factor=None, maxfev=np.inf, maxgev=np.inf, sigma=1.4901161193847656e-08, trials=1, verbose=False):
        self.objective = objective
        self.epsilon = epsilon
        self.batch_size = batch_size
        self.T_0 = T_0
        self.sigma = sigma # Default in scipy: https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.fmin_bfgs.html
        self.verbose = verbose
        self.f, self.grad_f, self.x_0 = get_objective(objective, n=n, m=m, nprob=nprob, factor=factor)
        self.name = name
        if objective is Objective.LOGISTIC:
            if nprob == 0: # Adult dataset
                self.num_points = 32561  # Used for selecting batches
                self.opt = 0.5264333233322221 # Found using CVXPY (within feastol=1.6e-10, reltol=7.1e-09, abstol=3.8e-09)
            elif nprob == 1: # Quantum dataset
                self.num_points = 50000  # Used for selecting batches
                self.opt = 0.3486057611952507 # Found using CVXPY (within feastol=2.5e-09, reltol=2.8e-08, abstol=9.9e-09)
            elif nprob == 2: # Protein dataset
                self.num_points = 145751  # Used for selecting batches
                self.opt = 0.6869838052545529 # Found using CVXPY (within feastol=3.1e-09, reltol=1.5e-08, abstol=1.0e-08)
        self.n = len(self.x_0)
        self.evals = 0
        self.grads = 0 # Counts iterations, which (in the non-DFO setting) correspond to gradient evaluations
        self.is_done = False
        self.final_val = np.inf
        self.final_x = None
        self.final_evals = 0
        self.maxfev = maxfev
        self.maxgev = maxgev
        self.trials = trials
        
    def val(self, x, increment_count=True, batch=None):
        if self.evals > self.maxfev:
            self.is_done = True
            return None
        if increment_count:
            self.evals = self.evals + 1
        return self.f(x, batch=batch)
    
    def grad(self, x, increment_count=True, batch=None, random=False):
        if increment_count:
            self.grads = self.grads + 1
            if self.objective is Objective.DFO:
                if random:
                    self.evals = self.evals + 1
                else:
                    self.evals = self.evals + self.n
        return self.grad_f(x, batch=batch, sigma=self.sigma, random=random)
    
    def reset(self, logfile=None):
        self.evals = 0
        self.grads = 0
        self.is_done = False
        self.final_val = np.inf
        self.final_x = None
        self.final_evals = 0
        self.logfile = logfile
        if self.logfile is not None:
            os.makedirs(os.path.dirname(self.logfile), exist_ok=True)
            with open(self.logfile, 'w') as f:
                f.write(f'0 {self.f(self.x_0)}\n')
        
    def done(self, x, increment_count=False, batch=None, is_callback=False):
        if increment_count:
            self.evals = self.evals + 1
        curval = self.f(x, batch=batch) # By default, check doneness on the full dataset
        if self.is_done:
            return True, curval  
        if self.logfile is not None:
            with open(self.logfile, 'a') as f:
                f.write(f'{self.evals} {curval}\n')
        if self.verbose:
            print(f'f(x)={"{:5.4}".format(curval)} and ||x||^2={"{:5.4}".format(np.dot(x, x))} after {self.grads} gradient evals and {self.evals} function evals')
        if is_callback: # scipy.optimize.minimize will stop if maxfev is exceeded already, so here we just need to check for accuracy
            self.grads = self.grads + 1 # done() gets called once per iteration
            self.final_evals = self.evals
            if self.epsilon is None:
                return
            if curval < self.epsilon:
                self.is_done = True
                self.final_x = x
                self.final_val = curval
                self.final_evals = self.evals  
        if self.evals >= self.maxfev or self.grads >= self.maxgev:
            self.is_done = True
            return True, curval
        if self.epsilon is None:
            return False, curval
        if self.objective is Objective.LOGISTIC: # Use relative error
            return (curval - self.opt) / self.opt < self.epsilon, curval
        return curval < self.epsilon, curval  

# Compute a Barzilai-Borwein initial step size
def BB(function):
    x0 = function.x_0
    g0 = function.grad(x0, increment_count=False)
    x1 = x0 - function.sigma * g0
    g1 = function.grad(x1, increment_count=False)
    return np.dot(g1 - g0, g1 - g0) / np.dot(x1 - x0, g1 - g0)


###############################################################################
# Implement the line search algorithms
###############################################################################

Algorithm = Enum('Algorithm', 'BACKTRACKING FORWARD_TRACKING APPROX_EXACT NELDER_MEAD WRIGHT_NOCEDAL BFGS FIXED')
globals().update(Algorithm.__members__)

class Method:
    def __init__(self, algorithm, random=False, momentum=False, bfgs=False, warm_start=True, beta=2 / (1 + 5 ** 0.5), name=None):
        self.algorithm = algorithm
        self.random = random
        self.momentum = momentum
        self.bfgs = bfgs
        assert(not (bfgs and momentum)) # BFGS and momentum are alternatives to steepest descent
        self.warm_start = warm_start
        self.beta = beta
        if algorithm.name is 'BACKTRACKING':
            if warm_start:
                self.name = 'ADAPTIVE_BACKTRACKING'
            else:
                self.name = 'TRADITIONAL_BACKTRACKING'
        else:
            self.name = algorithm.name
        self.label = self.name
        if name is not None:
            self.label = name

def find_stepsize_backtracking(f, T, beta, x, direction, grad, curval, batch=None):
    t = T
    grad_norm_squared = np.dot(grad, direction)
    f_old = curval
    f_new = f.val(x - t * direction, batch=batch)
    if f_new is None:
        return 0, f_old
    while f_new > curval - 0.5 * t * grad_norm_squared: # Backtrack
        t = beta * t  # Decrease the step size
        f_old = f_new
        f_new = f.val(x - t * direction, batch=batch)
        if f_new is None:
            return 0, curval
    return t, f_new

def find_stepsize_forward_tracking(f, T, beta, x, direction, grad, curval, batch=None):
    t = T
    t_old = 0
    grad_norm_squared = np.dot(grad, direction)
    f_old = curval
    f_new = f.val(x - t * direction, batch=batch)
    if f_new is None:
        return 0, f_old
    if f_new < curval - 0.5 * t * grad_norm_squared: # Forward-track
        while f_new < curval - 0.5 * t * grad_norm_squared:
            t_old = t
            t = t / beta  # Increase the step size
            f_old = f_new
            f_new = f.val(x - t * direction, batch=batch)
            if f_new is None:
                return t_old, f_old
        return t_old, f_old
    else: # Backtrack
        while f_new > curval - 0.5 * t * grad_norm_squared:
            t = beta * t  # Decrease the step size
            f_old = f_new
            f_new = f.val(x - t * direction, batch=batch)
            if f_new is None:
                return 0, curval
        return t, f_new

def find_stepsize_approx_exact(f, T, beta, x, direction, curval, batch=None):
    t = T
    t_old = 0
    f_old_old = curval
    f_old = curval
    f_new = f.val(x - t * direction, batch=batch)
    f_first = f_new
    if f_new is None:
        return 0, f_old
    param = beta
    if f_new <= f_old:
        param = 1.0 / beta
    num_iters = 0
    count = 0
    while True:
        num_iters = num_iters + 1
        t_old_old = t_old
        t_old = t
        t = t * param
        f_old_old = f_old
        f_old = f_new
        f_new = f.val(x - t * direction, batch=batch)
        if f_new is None:
            return 0, curval
        if f_new > curval and param < 1: # Special case for nonconvex functions to ensure function decrease
            continue
        if f_new == f_old: # Numerical precision can be a problem in flat places, so try increasing step size
            count = count + 1
        if count > 20: # But have limited patience for asymptotes/infima
            break
        if f_new > f_old or (f_new == f_old and param < 1):
            break
    # Handle special case where the function value decreased at t but increased at t/beta
    if count > 20 or (num_iters == 1 and param > 1):
        t = t_old # Ok to stop updating t_old, t_old_old, and f_old_old once we're backtracking
        param = beta
        f_old, f_new = f_new, f_old
        if count > 20: # Tried increasing step size, but function was flat
            t = T
            f_new = f_first
        count = 0
        while True:
            t = t * param
            f_old = f_new
            f_new = f.val(x - t * direction, batch=batch)
            if f_new is None:
                return 0, curval
            if f_new == f_old:
                count = count + 1
            if count > 20: # Don't backtrack forever if the function is flat
                break
            if f_new > f_old:
                break
    if param < 1:
        return t, f_new
    return t_old_old, f_old_old

def line_search(f, method, seed=0):
    if method.algorithm is Algorithm.NELDER_MEAD: 
        assert(f.batch_size is None) 
        if f.verbose:
            print(f'Initial function value is {f.val(f.x_0, increment_count=False)}')
        opt = scipy.optimize.minimize(f.val, f.x_0, method='Nelder-Mead', options={'maxfev': f.maxfev, 'xatol': 0.0, 'fatol': 0.0}, callback=f.done)
        if f.epsilon is None:
            return opt.x, opt.nit, None, np.asarray([opt.fun])
        return f.final_x, f.grads, None, np.asarray([f.final_val])
    if method.algorithm is Algorithm.BFGS:
        assert(f.batch_size is None) 
        if f.verbose:
            print(f'Initial function value is {f.val(f.x_0, increment_count=False)}')
        scipy.optimize.fmin_bfgs(f.val, f.x_0, gtol=-1., callback=f.done)
        return f.final_x, f.grads, None, np.asarray([f.final_val])
    step = 0
    losses = []
    x = f.x_0
    prev_prev_x = x
    prev_x = x
    np.random.seed(seed=seed)
    t = f.T_0
    t_prime = None
    batch = None
    time_checking_done = 0
    done = False
    j = 1
    kmin = 10
    grad = None
    if method.bfgs:
        B_inv = 1./f.T_0 * np.eye(len(x))
    while not done:
        step = step + 1
        # Pick a batch at random
        if f.batch_size is not None:
            batch = np.random.choice(f.num_points, size=f.batch_size, replace=False) 
        # Nesterov momentum
        y = x
        if method.momentum:
            y = x + (j-1)/(j+2) * (x - prev_x)
            curval = f.val(y, batch=batch)
        # Evaluate the function at the initial point
        elif method.algorithm is not Algorithm.FIXED and f.batch_size is not None:
            curval = f.val(y, batch=batch)
        elif method.algorithm is not Algorithm.FIXED and f.batch_size is None and step is 1:
            curval = f.val(y, batch=batch)
            if f.verbose:
                print(f'initial function value is {curval}')
        # Pick the descent direction
        old_grad = grad
        grad = f.grad(y, batch=batch, random=method.random)
        grad_norm_squared = np.dot(grad, grad)
        # BFGS direction
        if method.bfgs:
            if old_grad is not None and t_prime is not None and t_prime > 0: 
                sk = x - prev_x
                yk = grad - old_grad
                sktyk = np.dot(sk, yk)
                # Workaround from scipy: https://github.com/scipy/scipy/blob/v1.4.1/scipy/optimize/optimize.py#L872-L964
                try:
                    rhok = 1. / sktyk
                except ZeroDivisionError:
                    rhok = 1000.
                if np.isnan(rhok) or np.isinf(rhok):
                    rhok = 1000.
                B_invyk = np.dot(B_inv, yk)
                term_2 = (sktyk + np.dot(yk, B_invyk)) * np.outer(sk, sk) * (rhok**2)
                term_3 = (np.outer(B_invyk, sk) + np.outer(sk, B_invyk)) * rhok
                B_inv = B_inv + term_2 - term_3
            direction = np.dot(B_inv, grad)
        else: 
            direction = grad 
        t_prime = None
        if grad_norm_squared == 0 or np.isinf(grad_norm_squared):
            t_prime = 0
        # If we manage to pick an ascent direction, do steepest descent instead
        elif np.dot(grad, direction) <= 0: 
            print('Ascent direction replaced with steepest descent, resetting values')
            t = f.T_0
            B_inv = 1./f.T_0 * np.eye(len(x))
            direction = grad
        if t_prime is None:
            # Pick the initial step size
            if method.warm_start:
                t_prime = t / method.beta 
                # Reset for extremely flat places
                if np.dot(prev_x - x, prev_x - x) == 0: 
                    t_prime = f.T_0
                    B_inv = 1./f.T_0 * np.eye(len(x))
            else:
                t_prime = f.T_0
            # Compute the step size
            if method.algorithm is Algorithm.FIXED:
                t_prime = f.T_0
            elif method.algorithm is Algorithm.BACKTRACKING:
                t_prime, curval = find_stepsize_backtracking(f, t_prime, method.beta, y, direction, grad, curval, batch=batch)
            elif method.algorithm is Algorithm.FORWARD_TRACKING:
                t_prime, curval = find_stepsize_forward_tracking(f, t_prime, method.beta, y, direction, grad, curval, batch=batch)
            elif method.algorithm is Algorithm.APPROX_EXACT:
                t_prime, curval = find_stepsize_approx_exact(f, t_prime, method.beta, y, direction, curval, batch=batch)
            elif method.algorithm is Algorithm.WRIGHT_NOCEDAL:
                # function.val returns None when you run out of function evals, and this causes a TypeError
                try:
                    t_prime, _, temp_curval = scipy.optimize.linesearch.line_search_armijo(f.val, y, -1*direction, grad, curval, alpha0=t_prime)
                except TypeError:
                    done = True
                    t_prime = 0
                if t_prime is None:
                    t_prime = 0
                else:
                    curval = temp_curval
            else: print('Unrecognized algorithm')
        prev_prev_x = prev_x
        prev_x = x
        if t_prime > 0:
            t = t_prime
        x = y - t_prime * direction
        if method.momentum:
            v1 = x - prev_x
            v2 = prev_x - prev_prev_x
            if np.dot(v1, v1) < np.dot(v2, v2) and j >= kmin:
                j = 1
            else:
                j = j + 1
        # Check for doneness (on the full dataset)
        # Keep track of time so that algorithms are not "billed" for time checking doneness
        start = time.process_time()
        done, loss = f.done(x)
        end = time.process_time()
        time_checking_done = time_checking_done + end - start
        losses.append(loss)
    return x, step, time_checking_done, np.asarray(losses)


###############################################################################
# Utility functions to run the algorithms and generate performance and data profiles
###############################################################################

# Wrapper for F that only loads the data when get_F is called
class Function:
    def __init__(self, objective, params, epsilon=0.0001, maxfev=10000, maxgev=10000, sigma=1.4901161193847656e-08, verbose=False):
        self.objective = objective
        self.maxfev = maxfev
        self.maxgev = maxgev
        self.sigma = sigma
        self.verbose = verbose
        self.trials = 1
        self.nprob = int(params[0])
        if objective is Objective.DFO:
            self.n = int(params[1])
            self.m = int(params[2])
            self.s = params[3]
            self.epsilon = None
            self.name = 'problem' + str(self.nprob) + 'n' + str(self.n) + 'm' + str(self.m) + 's' + str(self.s) + 'epsilon' + str(self.epsilon)
        elif self.objective is Objective.LOGISTIC:
            self.T_0 = params[1]
            self.batch_size = int(params[2])
            if self.batch_size == 0: # Notation for full batch
                self.batch_size = None
            else:
                self.trials = 10
            self.epsilon = params[3]
            self.name = 'problem' + str(self.nprob) + 'batchsize' + str(self.batch_size) + 'T_0' + str(self.T_0) + 'epsilon' + str(self.epsilon)

    def get_F(self):
        if self.objective is Objective.DFO:
            # DFO termination criterion is maxfev
            return F(self.objective, epsilon=self.epsilon, n=self.n, m=self.m, nprob=self.nprob, factor=10**self.s, maxfev=self.maxfev, sigma=self.sigma, verbose=self.verbose)
        elif self.objective is Objective.LOGISTIC:
            # Logistic regression termination criterion is generally epsilon, but maxgev is included in case an algorithm is too slow
            return F(self.objective, epsilon=self.epsilon, nprob=self.nprob, batch_size=self.batch_size, T_0=self.T_0, maxgev=self.maxgev, sigma=self.sigma, trials=self.trials, verbose=self.verbose)

def get_functions(objective, probs_file, maxfev=10000, maxgev=10000, sigma=1.4901161193847656e-08, verbose=False):
    functions = []
    probs = np.loadtxt(probs_file)
    for i in range(len(probs)):
        params = probs[i]
        functions.append(Function(objective=objective, params=params, maxfev=maxfev, maxgev=maxgev, sigma=sigma, verbose=verbose))
    return np.asarray(functions)

def prep_prof(function, method, trial=0, filename_prefix='./tmp_output/', reuse=True):
    f = function.get_F()
    prefix = filename_prefix + function.objective.name + '/'
    logfile = prefix + method.label + '/' + function.name + 'trial' + str(trial) + '.txt'
    if not reuse or not os.path.exists(logfile):
        f.reset(logfile=logfile)
        start = time.process_time()
        _, _, time_checking_done, losses = line_search(f, method, seed=trial)
        end = time.process_time()
        # Last line of logistic logfile contains total process time and final accuracy
        if function.objective is Objective.LOGISTIC:
            with open(logfile, 'a') as f:
                f.write(f'{end - start - time_checking_done} {losses[-1]}\n')
    return logfile

def prepare_prof(functions, methods, tau=0.001, filename_prefix='./tmp_output/', reuse=True):
    opt_vals = []
    method_names = []
    for method in methods:
        method_names.append(method.label)
    for function in functions:
        for trial in range(function.trials):
            if function.objective is Objective.DFO:
                all_data = []
                min_loss = np.inf
            else:
                times = []
            for method in methods: 
                logfile = prep_prof(function, method, trial=trial, filename_prefix=filename_prefix, reuse=reuse)
                bad = check(np.asarray([function]), np.asarray([method]))
                bad = (len(bad) > 0)
                data = np.loadtxt(logfile)
                if function.objective is Objective.DFO:
                    opt_val = data[data[:, 0] <= function.maxfev/10., 1][-1] # Take the last line as the best value achieved
                    if opt_val < min_loss:
                        min_loss = opt_val
                    all_data.append(data)
                else:
                    if bad:
                        times.append(np.inf)
                    else:
                        times.append(data[-1, 0]) # Logistic objective is concerned with process time
            if function.objective is Objective.DFO:
                all_data = np.asarray(all_data)
                # First line of log files is initial loss
                f_0 = all_data[0][0,1] 
                epsilon = min_loss + tau * (f_0 - min_loss)
                # For each method, find the minimum number of function evaluations needed to achieve loss < epsilon
                evals = []
                for i in range(len(methods)):
                    indices = all_data[i][:,1] < epsilon
                    if np.sum(indices) > 0:
                        evals.append(all_data[i][indices, 0][0])
                    else:
                        evals.append(function.maxfev)
                opt_vals.append(np.asarray(evals))
            else:
                opt_vals.append(np.asarray(times))
    opt_vals = np.asarray(opt_vals)
    return opt_vals, method_names

def data_prof(functions, methods, filename_prefix='./tmp_output/', reuse=True, names=None, filename=None, fullsize=False, legend=True):
    opt_vals, method_names = prepare_prof(functions=functions, methods=methods, filename_prefix=filename_prefix, reuse=reuse)
    if names is not None:
        method_names = names
    tau_min = 1.0
    npts = 500
    num_prob, num_method = opt_vals.shape
    rho = np.zeros((npts, num_method))
    # Compute the tau range to consider
    min_n = np.inf
    for prob in range(num_prob):
        min_n = min(min_n, functions[prob].n)
    tau_max = min(100, 1.1 * np.max(np.max(opt_vals, axis=1) / (min_n + 1)))
    # Compute the data profile
    tau = np.linspace(tau_min, tau_max, npts)
    for method in range(num_method):
        for k in range(npts):
            total = 0
            for prob in range(num_prob):
                total = total + (opt_vals[prob, method] / (functions[prob].n + 1) <= tau[k])
            rho[k,method] = total / num_prob
    plot_prof(tau=tau, rho=rho, method_names=method_names, linestyles=['-', '--', '-.', ':'], filename=filename, x_label='Number of Simplex Gradients', fullsize=fullsize, legend=legend)

def perf_prof(functions, methods, filename_prefix='./tmp_output/', reuse=True, names=None, filename=None, fullsize=False, legend=True):
    opt_vals, method_names = prepare_prof(functions=functions, methods=methods, filename_prefix=filename_prefix, reuse=reuse)
    print(method_names)
    print(opt_vals)
    if names is not None:
        method_names = names
    tau_min = 1.0
    npts = 100
    minval = 1e-100
    num_prob, num_method = opt_vals.shape
    rho = np.zeros((npts, num_method))
    # Compute the tau range to consider
    tau_max = min(100, 1.1 * np.max(np.max(opt_vals, axis=1) / (np.min(opt_vals, axis=1) + minval)))
    # Compute the cumulative rates of the performance ratio being at most a fixed threshold
    tau = np.linspace(tau_min, tau_max, npts)
    for method in range(num_method):
        for k in range(npts):
            rho[k, method] = np.sum(opt_vals[:, method] / (np.min(opt_vals, axis=1) + minval) <= tau[k]) / num_prob
    plot_prof(tau=tau, rho=rho, method_names=method_names, linestyles=['-', '--', '-.', ':'], filename=filename, x_label='Performance Ratio', fullsize=fullsize, legend=legend)

def plot_prof(tau, rho, method_names, linestyles, filename, x_label, fullsize=False, legend=True):
    label_fontsize = 20
    tick_fontsize = 20
    linewidth = 3
    if fullsize:
        linewidth = 4
        plt.figure(figsize=(15,10))
    else:
        plt.figure(figsize=(9,6))
    colors = [ '#2D328F', '#F15C19',"#81b13c","#ca49ac","000000"] 
    if len(colors) < len(method_names):
        difference = len(method_names) - len(colors)
        colors = [colors[0]] * (difference + 1) + colors[1:]
    if len(linestyles) + 3 == len(method_names): # Linestyles for DFO plot
        linestyles = [linestyles[0]] + [(0, (1,2,3,2,1,2)), (0, (1,1))] + linestyles[1:] + [(0, (3,1,1,1,1,1))]
    else:
        linestyles = linestyles * len(method_names)
    # make plot
    for method in range(len(method_names)):
        plt.plot(tau, rho[:,method], color=colors[method], linestyle=linestyles[method], linewidth=linewidth,label=method_names[method], alpha=0.9)
        
    plt.xlabel(x_label,fontsize=label_fontsize)
    if legend:
        plt.legend(fontsize=label_fontsize)
    plt.xticks(fontsize=tick_fontsize)
    plt.yticks(fontsize=tick_fontsize)
    plt.grid()
    plt.tight_layout()
    plt.show()
    if filename is not None:
        plt.savefig(filename, format='pdf')

# Helper function for checking which experiments are still running
# and removing unfinished log files
def check(functions, methods, clean=False, verbose=False):
    bad = []
    filename_prefix='./tmp_output/'
    prefix = filename_prefix + functions[0].objective.name + '/'
    for function in functions:
        for method in methods: 
            for trial in range(function.trials):
                logfile = prefix + method.label + '/' + function.name + 'trial' + str(trial) + '.txt'
                if not os.path.exists(logfile):
                    if verbose:
                        print(f'{logfile} is missing')
                    bad.append(logfile)
                    continue
                data = np.loadtxt(logfile)
                if len(data.shape) < 2:
                    if verbose:
                        print(f'{logfile} only has one step')
                    bad.append(logfile)
                    if clean:
                        os.remove(logfile)
                    continue
                if function.objective is Objective.LOGISTIC:
                    endtime = data[-1, 0]
                    if endtime - int(endtime) == 0:
                        if verbose:
                            print(f'{logfile} is not done')
                        bad.append(logfile)
                        if clean:
                            os.remove(logfile)
                if function.objective is Objective.DFO:
                    evals = data[-1, 0]
                    if evals < 0.99 * function.maxfev:
                        if verbose:
                            print(f'{logfile} is not done')
                        bad.append(logfile)
                        if clean:
                            os.remove(logfile)
    return np.asarray(bad)
