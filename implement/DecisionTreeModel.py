import sys
sys.path.append('..')

from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_graphviz
from utils import load_iris

class DecisionTreeModel:
    
    def __init__(self,depth = 2):
        self.tree = DecisionTreeClassifier(max_depth=depth)
        self.iris = load_iris()
    
    def c_demo(self):
        X = self.iris.data[:, 2:]
        Y = self.iris.target
        self.classifier(X, Y, True)

    def classifier(self, x, y, need_show=False):
        self.tree.fit(x, y)
        if need_show:
            export_graphviz(
                self.tree,
                out_file="iris_tree.dot",
                feature_names=self.iris.feature_names[2:],
                class_names=self.iris.target_names,
                rounded=True,
                filled=True
            )    

    def predict(self, sample):
        return self.tree.predict_proba(sample)
