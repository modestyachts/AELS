# AELS
Approximately Exact Line Search

## Files
`utils.py` contains the implementation of Approximately Exact Line Search, as well as various baseline methods and helper functions.  

`profile.py` runs the line search methods on the benchmark problems, and saves the results. Options are:  
        `--problems_file`: the `.txt` file describing the desired problems to run  
        `--objective`: either `dfo` or `logistic`  
        `--recompute`: pass this option to force recomputation (prior results are reused by default)  
        `--threads`: number of processes to run concurrently; for accurate timing data, we recommend fewer processes than cores  
        `--verbose`: pass this option to print progress details (recommend piping to a file)  
        
`make_profiles.py` generates the figures in our paper. We recommend running `profile.py` before `make_profiles.py`. Options are:  
        `--type`: `dfo` (benchmark DFO problems), `batchsize` (logistic regression with various minibatch sizes), or `epsilon` (logistic regression with various required accuracies)  
        `--threads`: number of processes to run concurrently  
        
`check.py` can be run concurrently with `profile.py` to monitor progress. `check.py` prints which experiments are in progress or remain to be initiated. Options are:  
        `--problems_file`: the `.txt` file describing the desired problems to run  
        `--objective`: either `dfo` or `logistic`  
        `--clean`: pass this option to delete logfiles for any unfinished experiments  
        `--quiet`: pass this option to print only logfile names for unfinished experiments

`problems/` contains `.txt` files describing the benchmark problems.

`data/` contains logistic regression benchmark data from the UCI Adult dataset (https://archive.ics.uci.edu/ml/datasets/Adult) and the KDD-Cup 2004 Physics and Biology datasets (https://www.kdd.org/kdd-cup/view/kdd-cup-2004/Tasks).  

`calfun.py`, `dfovec.py`, and `dfoxs.py` contain Python implementations of the DFO benchmark problems at https://www.mcs.anl.gov/~more/dfo/. 

## Example
To run the DFO experiments and generate logfiles:  
`python profile.py --problems_file=problems/dfo.txt --objective=dfo`

While that is running, in another window you can monitor progress by running:  
`python check.py --problems_file=problems/dfo.txt --objective=dfo`

To plot the performance profile:  
`python make_profiles.py --type=dfo`

## Paper
This repository contains the code to reproduce results in the following paper:  
https://arxiv.org/abs/2011.04721
