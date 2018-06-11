import os
import pickle
import numpy as np
import sklearn.datasets as datasets
import matplotlib.pyplot as plt

from scipy import io
from sklearn.model_selection import StratifiedKFold

# from sklearn.datasets import fetch_mldata
# MLDATA_BASE_URL = "http://mldata.org/repository/data/"

_data_path = '_data/'
_model_path = '_model/'
_post = '.pkl'
_kfold = 5
_seed = 44

def load_iris(return_xy=False):
    iris = datasets.load_iris()
    if return_xy:
        X, Y = iris['data'], iris['target']
        return X, Y
    return iris

def load_mnist(return_xy=False):
    mnist = load_mat()
    if return_xy:
        X, Y = mnist['data'], mnist['label']
        return X, Y
    return mnist

def load_mat(fname='mnist-original.mat'):
    # mnist return dict{'data': (784, 70000), 'label': (1, 70000)}
    # load matlab file
    with open((_data_path + fname), 'rb') as matlab_file:
        matlab_dict = io.loadmat(matlab_file, struct_as_record=True)
    return matlab_dict
    
def load_csv(fname=''):
    with open((_data_path + fname), 'rb') as f:
        pass
    
def show_digit(arr, label, re_w, re_h, _show=False):
    '''
       arr from mnist['data'].T
       label from mnist['label'].T
       e.g. show_digit(X[36000], Y[36000], 28, 28)
    '''
    digit_img = arr.reshape(re_w, re_h)
    plt.imshow(digit_img, cmap=plt.cm.binary, interpolation='nearest')
    plt.axis('off')
    plt.title(label)
    if _show:
        plt.show()

def plot_x_y(x, y, x_min, x_max, y_min, y_max, kind='b.', label='', loc="lower right", _show=False):
    plt.plot(x, y, kind, label=label)
    plt.ylabel("Y")
    plt.xlabel("X")
    plt.xlim([x_min, x_max])
    plt.ylim([y_min, y_max])
    plt.legend(loc=loc)
    if _show:
        plt.show()

def divide(X, Y, need_shuffle=False):
    pct = (_kfold - 1)/_kfold
    def _shuffle(X, Y):
        idx = np.random.permutation(len(X))
        return X[idx], Y[idx]
    if need_shuffle:
        X, Y = _shuffle(X, Y)
    cut = int(len(X) * pct)
    x_tr, x_ts = X[:cut], X[cut:]
    y_tr, y_ts = Y[:cut], Y[cut:]
    return (x_tr, y_tr), (x_ts, y_ts)

def kFold_div(x, y, need_shuffle=False):
    skfolds = StratifiedKFold(n_splits=_kfold, random_state=_seed, shuffle=need_shuffle)
    for tr_idx, ts_idx in skfolds.split(x, y):
        yield (x[tr_idx], y[tr_idx]), (x[ts_idx], y[ts_idx])

def save_model(model, mname):
    with open(_model_path + mname + _post, 'wb') as f:
        pickle.dump(model, f)

def load_model(mname):
    if not os.path.isfile(_model_path + mname + _post):
        raise Exception('ERROR: {} does not exist here'.format(_model_path + mname + _post))
    with open(_model_path + mname + _post, 'rb') as f:
        return pickle.load(f)

mnist = load_mnist()
X, Y = mnist['data'].T, mnist['label'].T.ravel()
(x_tr, y_tr), (x_ts, y_ts) = divide(X, Y)
# g = kFold_div(X, Y)
# show_digit(X[36000], Y[36000], 28, 28)

