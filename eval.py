import warnings
warnings.filterwarnings("ignore")
import numpy as np 
import matplotlib.pyplot as plt
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import RandomForestClassifier
from implement.LinearModel import LinearModel
from validation import Validation
from utils import x_tr, y_tr, x_ts, y_ts, save_model, load_model, load_iris

try:
    sgd = load_model('sgd')
except Exception as e:
    sgd = SGDClassifier(random_state=42)
rfc = RandomForestClassifier(n_estimators=10)

val = Validation()

y_tr_5 = (y_tr == 5)
y_ts_5 = (y_ts == 5)

# y_pred, y_score = val.cross_val(sgd, x_tr, y_tr_5)
# conf_mat, precision, recall, f1 = val.scored(y_tr_5, y_pred)

# print('cross val score: ', y_score)
# print('confusion matrix: \n', conf_mat)
# print('precision: ', precision)
# print('recall: ', recall)
# print('F1: ', f1)

# # show confusion matrix
# val.plot_conf_matrix(conf_mat, _show=True)

# # show sgd precision-recall curve
# y_pred, _ = val.cross_val(sgd, x_tr, y_tr_5, method='decision_function')
# precisions, recalls, thresholds = val.precision_recall_analyse(y_tr_5, y_pred)
# val.plot_precision_recall(precisions, recalls, _show=True)
# val.plot_precision_recall_vs_threshold(precisions, recalls, thresholds, _show=True)

# # show sgd roc curve
# fpr, tpr, thresholds, auc = val.roc_auc_analyse(y_tr_5, y_pred)
# val.plot_roc(fpr, tpr, auc=auc, label='SGD')

# # show random forest roc curve
# y_pred, _ = val_rf.cross_val(rfc, x_tr, y_tr_5, method='predict_proba')
# fpr, tpr, thresholds, auc = val_rf.roc_auc_analyse(y_tr_5, y_pred[:,1])
# val_rf.plot_roc(fpr, tpr, kind='g--', auc=auc, label='RF', _show=True)

# save_model(sgd, 'sgd')



# lrm = LinearModel()
# iris = load_iris()
# X = iris["data"][:, 3:]
# y = (iris["target"] == 2)
# lrm.lgsr.fit(X, y)
# X_new = np.linspace(0, 3, 1000).reshape(-1, 1)
# y_proba = lrm.lgsr.predict_proba(X_new)
# plt.plot(X_new, y_proba[:, 1], "g-", label="Iris-Virginica")
# plt.plot(X_new, y_proba[:, 0], "b--", label="Not Iris-Virginica")
# plt.show()

# mlr.lr_demo()
# mlr.poly_demo()

# X, X_b, y, theta = mlr.demo(need_return=True)
# mlr.ridge.fit(X, y)
# print(mlr.ridge.predict([[1.5]]))
# mlr.sgdr.fit(X, y.ravel())
# print(mlr.sgdr.predict([[1.5]]))
# mlr.lasso.fit(X, y)
# print(mlr.lasso.predict([[1.5]]))
# mlr.elast.fit(X, y)
# print(mlr.elast.predict([[1.5]]))