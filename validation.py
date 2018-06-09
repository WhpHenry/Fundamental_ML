import matplotlib.pyplot as plt

from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import cross_val_score
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix
# from sklearn.metrics import accuracy_score


class Validation:
    def __init__(self, model, kfolds=10):
        self.model = model
        self.kfolds = kfolds
    
    def cross_val(self, x, y, method="predict", scoring='accuracy'):
        '''
        method:  e.g. accuracy / decision_fucntion / predict_proba
        scoring: e.g. accuracy / decision_fucntion; more on:
        http://scikit-learn.org/stable/modules/model_evaluation.html#scoring-parameter
        '''
        y_pred = cross_val_predict(self.model, x, y, cv=self.kfolds, method=method)
        y_score = cross_val_score(self.model, x, y, cv=self.kfolds, scoring=scoring)
        return y_pred, y_score
    
    def scored(self, y_true, y_pred):
        conf_mat = confusion_matrix(y_true, y_pred)
        precision = precision_score(y_true, y_pred)
        recall = recall_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)
        return conf_mat, precision, recall, f1
    
    def precision_recall_analyse(self, y, y_pred):
        '''
        y_pred from self.cross_val
        '''
        precisions, recalls, thresholds = precision_recall_curve(y, y_pred)
        return precisions, recalls, thresholds

    def roc_auc_analyse(self, y, y_pred):
        '''
        y_pred from self.cross_val
        '''
        fpr, tpr, thresholds = roc_curve(y, y_pred)
        auc = roc_auc_score(y, y_pred)
        return fpr, tpr, thresholds, auc

    def plot_conf_matrix(self, conf_mat, cmap=plt.cm.gray, _show=False, _norm=False):
        if _norm:
            row_sums = conf_mat.sum(axis=1, keepdims=True)
            conf_mat = conf_mat / row_sums
        plt.matshow(conf_mat, cmap=cmap)
        if _show:
            plt.show()

    def plot_precision_recall_vs_threshold(self, precisions, recalls, thresholds, _show=False):
        '''
        show recall curve and precision curve vs threshold
        '''
        plt.plot(thresholds, precisions[:-1], "b--", label="Precision")
        plt.plot(thresholds, recalls[:-1], "g-", label="Recall")
        plt.xlabel("Threshold")
        plt.legend(loc="upper left")
        plt.ylim([0, 1])
        if _show:
            plt.show()

    def plot_precision_recall(self, precisions, recalls, _show=False):
        '''
        show recall - precision curve
        '''
        plt.plot(recalls, precisions, "b-")
        plt.ylabel("Precision")
        plt.xlabel("Recall")
        plt.ylim([0, 1])
        if _show:
            plt.show()

    def plot_fpr_tpr_threshold(self, fpr, tpr, thresholds, _show=False):
        plt.plot(thresholds, tpr, "b--", label="tpr")
        plt.plot(thresholds, fpr, "g-", label="fpr")
        plt.xlabel("Threshold")
        plt.legend(loc="upper left")
        plt.ylim([0, 1])
        if _show:
            plt.show()

    def plot_roc(self, fpr, tpr, kind='b-', auc=None, label='SGD', _show=False):
        '''
        show roc curve
        '''
        if auc:
            label += (' - AUC: ' + str(auc))
        plt.plot(fpr, tpr, kind, label=label)
        plt.plot([0, 1], [0, 1], 'k--')
        plt.axis([0, 1, 0, 1])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.legend(loc="lower right")
        if _show:
            plt.show()
