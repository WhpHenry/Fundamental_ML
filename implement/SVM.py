import sys
sys.path.append('..')

import numpy as np
from sklearn.svm import LinearSVC
from whp_decorate import func_cost
from utils import plot_x_y, load_iris