import numpy as np
from classes.TreeGB import TreeGradientBoosting


class GradientBoosting:
    def __init__(self, max_features=-1, min_samples_split=2, max_depth=5, n_estimators=20, threshold=0.5,
                 learning_rate=0.1, regression=True):
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.max_features = max_features
        self.max_depth = max_depth
        self.n_estimators = n_estimators
        self.threshold = threshold
        self.lr = learning_rate
        self.weak_learners = []
        self.regression = regression

    def fit(self, X, y):
        # https://scikit-learn.org/stable/modules/ensemble.html#gradient-boosting
        n_sample, n_feats = X.shape
        self.max_features = n_feats if self.max_features == -1 else self.max_features
        # Step 1:
        if self.regression:
            self._mean_train = np.mean(y)
            y_pred = np.full_like(y, fill_value=self._mean_train)
        else:
            # odds
            odds = np.bincount(y)
            log_odds = np.full(y.shape, np.log(odds[0] / odds[1]))
            y_pred = self.logistic_function(log_odds)
        global previous_proba
        previous_proba = y_pred.copy()

        for _ in range(self.n_estimators):
            res = (y - y_pred) * 2
            # Step 2:
            tree = TreeGradientBoosting(max_depth=self.max_depth, max_features=self.max_features,
                                        min_samples_split=self.min_samples_split, regression=self.regression)

            # tree = DecisionTreeRegressor(max_depth=self.max_depth, min_samples_split = self.min_samples_split)
            tree.fit(X, res, **{'previous_proba':previous_proba})
            # Step 3:
            if not self.regression:
                log_odds += np.array(tree.predict(X)) * self.lr
                y_pred = self.logistic_function(log_odds)
            else:
                new = tree.predict(X)
                y_pred = np.add(y_pred, np.array(new).reshape(len(new),1) * self.lr)
            previous_proba = y_pred.copy()
            self.weak_learners.append(tree)

    @staticmethod
    def logistic_function(x):
        return np.exp(x) / (1 + np.exp(x))

    def predict(self, X):
        """
        Not enough tree will lead to every prediction less than a common threshold such as 0.5
        So a trade off has to be made between n_estimators (tree) and the learning rate
        """
        if not self.regression:
            predictions = np.zeros(len(X))
            for tree in self.weak_learners:
                predictions += self.lr * np.array(tree.predict(X))

            return predictions  # np.where(predictions > self.threshold, 1, 0)

        else:
            pred = np.ones((X.shape[0],)) * self._mean_train
            for tree in self.weak_learners:
                pred = np.add(pred, self.lr * np.array(tree.predict(X)))

            return pred


