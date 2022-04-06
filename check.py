import os
os.environ['OMP_NUM_THREADS'] = '1'
import numpy as np
from argparse import ArgumentParser
from utils import *
from definitions import *

parser = ArgumentParser()
parser.add_argument('--problems_file', type=str, default='dfo.txt', help='name of file specifying benchmark problems')
parser.add_argument('--objective', type=Objective, default='dfo', choices=list(Objective))
parser.add_argument('--clean', action='store_true', help='delete the unfinished logfiles')
parser.add_argument('--quiet', action='store_true', help='don\'t print what is wrong with each unfinished experiment')
args= parser.parse_args()

if args.objective is Objective.LOGISTIC:
    algs = np.asarray([ae, ba, tba, wolfe_cheap, co, inv])
else: 
    algs = np.asarray([aeb, bab, tbab, nm, bfgs_cheap])

funcs = get_functions(args.objective, args.problems_file)

if __name__ == "__main__":
    check(funcs, algs, clean=args.clean, verbose=not args.quiet)