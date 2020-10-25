import numpy as np
import sys
import os
import multiprocessing
from argparse import ArgumentParser
from utils import *
from definitions import *

class Kind(Enum):
    DFO = 'dfo'
    BATCHSIZE = 'batchsize'
    EPSILON = 'epsilon'

    def __str__(self):
        return self.value

parser = ArgumentParser()
parser.add_argument('--threads', type=int, default=1, help='number of processes to run concurrently')
parser.add_argument('--type', type=Kind, default='batchsize', choices=list(Kind))
args= parser.parse_args()

prefix = './figures/'
if args.type is Kind.DFO:
    algs = np.asarray([aeb, ae, aer, bab, tbab, nm, bfgs])
    probs_files = np.asarray(['problems/dfo.txt'])
    objective = Objective.DFO
elif args.type is Kind.BATCHSIZE:
    algs = np.asarray([ae, ba, tba])
    probs_files = np.asarray(['problems/logistic_sgd_batchsize_None.txt', 
                            'problems/logistic_sgd_batchsize_1600.txt',
                            'problems/logistic_sgd_batchsize_400.txt',
                            'problems/logistic_sgd_batchsize_100.txt',
                            'problems/logistic_sgd_batchsize_25.txt',
                            'problems/logistic_sgd_batchsize_6.txt',
                            'problems/logistic_sgd_batchsize_1.txt'])
    objective = Objective.LOGISTIC
elif args.type is Kind.EPSILON:
    algs = np.asarray([ae, ba, tba])
    probs_files = np.asarray(['problems/logistic_sgd_epsilon_pt001.txt', 
                              'problems/logistic_sgd_epsilon_pt0001.txt',
                              'problems/logistic_sgd_epsilon_pt00001.txt',
                              'problems/logistic_sgd_epsilon_pt000001.txt'])
    objective = Objective.LOGISTIC


def perf_func(functions, algs, filename, fullsize, legend):
    perf_prof(functions, algs, filename=filename, fullsize=fullsize, legend=legend)

def data_func(functions, algs, filename, fullsize, legend):
    data_prof(functions, algs, filename=filename, fullsize=fullsize, legend=legend)

pool_size = multiprocessing.cpu_count()
os.system('taskset -cp 0-%d %s' % (pool_size, os.getpid()))

pool = multiprocessing.Pool(args.threads) 
for probs_file in probs_files:
    functions = get_functions(objective, probs_file)
    if probs_file == probs_files[0] and args.type is not Kind.EPSILON:
        pool.apply_async(perf_func, args=(functions, algs, prefix + probs_file[0:-4] + '.pdf', True, True))
        if args.type is Kind.DFO:
            pool.apply_async(data_func, args=(functions, algs, prefix + probs_file[0:-4] + '_data.pdf', True, True))
    elif probs_file == probs_files[-1]:
        pool.apply_async(perf_func, args=(functions, algs, prefix + probs_file[0:-4] + '.pdf', False, True))
    else:
        pool.apply_async(perf_func, args=(functions, algs, prefix + probs_file[0:-4] + '.pdf', False, False))
pool.close()
pool.join()

