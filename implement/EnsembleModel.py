from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier 
from sklearn.ensemble import VotingClassifier 
from sklearn.ensemble import BaggingClassifier 
from sklearn.tree import DecisionTreeRegressor 
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression 
from sklearn.svm import SVC

from sklearn.metrics import accuracy_score


class EnsembleModel:
    def __init__(self):
        self.basic_model()
    
    def demo_enesable(self, x_tr, x_ts, y_tr, y_ts):
        model = self.vote
        model.fit(x_tr, y_tr)
        y_pred = model.predict(x_ts)
        print(model.__class__.__name__, accuracy_score(y_ts, y_pred))

    def build_vote(self, voting='hard', weights=None):
        return VotingClassifier(
                    estimators = [('lgr', self.lgr), ('rf', self.rnd), ('svc', self.svm)],
                    voting = voting,
                    weights = weights
                )

    def build_bag(self, num_units=500, samples=100):
        return BaggingClassifier(
                    self.tree, 
                    n_estimators=num_units,
                    max_samples=samples,
                    bootstrap=True,
                    n_jobs=-1
                ) 
    
    def build_forest(self, num_units=500, leaves=16):
        return RandomForestClassifier(n_estimators=num_units, max_leaf_nodes=leaves, n_jobs=-1) 

    def build_adaboost(self, num_units=200, learning_rate=0.5):
        return AdaBoostClassifier(
                    DecisionTreeClassifier(max_depth=1), 
                    n_estimators=num_units,
                    algorithm="SAMME.R", 
                    learning_rate=learning_rate
                ) 
    
    def build_gbdt(self, depth=2, num_units=3, learning_rate=1.0):
        '''
        gbdt basicly same as follow:
        tr1 = DecisionTreeRegressor(max_depth=2) 
        tr2 = DecisionTreeRegressor(max_depth=2) 
        tr3 = DecisionTreeRegressor(max_depth=2) 

        tr1.fit(x, y)
        y2 = y - tr1.predict(x)
        tr2.fit(x, y2)
        y3 = y2 - tr2.predict(x)
        tr3.fit(x, y3)

        y_pred = sum(tree.predict(x_ts) for tree in (tr1, tr2, tr3))
        '''

        return GradientBoostingRegressor(
                    max_depth=depth, 
                    n_estimators=num_units, 
                    learning_rate=learning_rate
                ) 


    def basic_model(self):
        self.lgr = LogisticRegression() 
        self.rnd = RandomForestClassifier() 
        self.svm = SVC()
        self.tree = DecisionTreeClassifier()