import os
os.environ['OMP_NUM_THREADS'] = '1'
import numpy as np
import sys
import multiprocessing
from argparse import ArgumentParser
from utils import *
from definitions import *
import time

start = time.time()

parser = ArgumentParser()
parser.add_argument('--problems_file', type=str, default='dfo.txt', help='name of file specifying benchmark problems')
parser.add_argument('--objective', type=Objective, default='dfo', choices=list(Objective))
parser.add_argument('--recompute', action='store_true', help='recompute instead of using stored data')
parser.add_argument('--threads', type=int, default=1, help='number of processes to run concurrently')
parser.add_argument('--verbose', action='store_true', help='print progress details')
args= parser.parse_args()

probs_file = args.problems_file
objective = args.objective
functions = get_functions(objective, probs_file, verbose=args.verbose)
reuse = not args.recompute
threads = args.threads

if objective is Objective.LOGISTIC:
    algs = np.asarray([ae, ba, tba])
else: 
    algs = np.asarray([aeb, ae, aer, bab, tbab, nm, bfgs])

def run_alg(function, alg, trial):
    prep_prof(function, alg, trial=trial, reuse=reuse)

pool_size = multiprocessing.cpu_count()
os.system('taskset -cp 0-%d %s' % (pool_size, os.getpid()))

pool = multiprocessing.Pool(threads) 
for alg in algs:
    for function in functions:
        for trial in range(function.trials):
            pool.apply_async(run_alg, args=(function, alg, trial))
pool.close()
pool.join()

end = time.time()
print(f'total time was {end - start}')