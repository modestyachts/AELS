from utils import Method, Algorithm

# Nelder-Mead
nm = Method(algorithm=Algorithm.NELDER_MEAD, name='Nelder-Mead')

# Wolfe BFGS
bfgs_cheap = Method(algorithm=Algorithm.BFGS, name='Wolfe + BFGS')
bfgs_expensive = Method(algorithm=Algorithm.BFGS, name='Expensive Wolfe + BFGS', cheap=False)
scipy_bfgs = Method(algorithm=Algorithm.SCIPY_BFGS, name='Scipy Wolfe + BFGS')

# Wolfe
wolfe_cheap = Method(algorithm=Algorithm.WOLFE, name='Wolfe')
wolfe_expensive = Method(algorithm=Algorithm.WOLFE, name='Expensive Wolfe', cheap=False)

# Wright-Nocedal line search
wn = Method(algorithm=Algorithm.WRIGHT_NOCEDAL, name='Wright-Nocedal')
wnr = Method(algorithm=Algorithm.WRIGHT_NOCEDAL, random=True, name='Wright-Nocedal + Random')
wnm = Method(algorithm=Algorithm.WRIGHT_NOCEDAL, momentum=True, name='Wright-Nocedal + Momentum')
wnb = Method(algorithm=Algorithm.WRIGHT_NOCEDAL, bfgs=True, name='Wright-Nocedal + BFGS')

# Approximately exact line search
ae = Method(algorithm=Algorithm.APPROX_EXACT, name='Approximately Exact')
aer = Method(algorithm=Algorithm.APPROX_EXACT, random=True, name='Approximately Exact + Random')
aem = Method(algorithm=Algorithm.APPROX_EXACT, momentum=True, name='Approximately Exact + Momentum')
aeb = Method(algorithm=Algorithm.APPROX_EXACT, bfgs=True, name='Approximately Exact + BFGS')

# Forward-tracking line search
ft = Method(algorithm=Algorithm.FORWARD_TRACKING, name='Forward-Tracking')
ftr = Method(algorithm=Algorithm.FORWARD_TRACKING, random=True, name='Forward-Tracking + Random')
ftm = Method(algorithm=Algorithm.FORWARD_TRACKING, momentum=True, name='Forward-Tracking + Momentum')
ftb = Method(algorithm=Algorithm.FORWARD_TRACKING, bfgs=True, name='Forward-Tracking + BFGS')

# Backtracking line search
ba = Method(algorithm=Algorithm.BACKTRACKING, name='Adaptive Backtracking')
bar = Method(algorithm=Algorithm.BACKTRACKING, random=True, name='Adaptive Backtracking + Random')
bam = Method(algorithm=Algorithm.BACKTRACKING, momentum=True, name='Adaptive Backtracking + Momentum')
bab = Method(algorithm=Algorithm.BACKTRACKING, bfgs=True, name='Adaptive Backtracking + BFGS')

# Traditional Backtracking line search
tba = Method(algorithm=Algorithm.BACKTRACKING, warm_start=False, name='Traditional Backtracking')
tbar = Method(algorithm=Algorithm.BACKTRACKING, random=True, warm_start=False, name='Traditional Backtracking + Random')
tbam = Method(algorithm=Algorithm.BACKTRACKING, warm_start=False, momentum=True, name='Traditional Backtracking + Momentum')
tbab = Method(algorithm=Algorithm.BACKTRACKING, warm_start=False, bfgs=True, name='Traditional Backtracking + BFGS')

# Constant step sizes
co = Method(algorithm=Algorithm.FIXED, warm_start=False, name='Constant')
cob = Method(algorithm=Algorithm.FIXED, warm_start=False, name='Constant + BFGS')

# Inverse step sizes
inv = Method(algorithm=Algorithm.INVERSE, warm_start=False, name='Inverse')
inb = Method(algorithm=Algorithm.INVERSE, warm_start=False, name='Inverse + BFGS')