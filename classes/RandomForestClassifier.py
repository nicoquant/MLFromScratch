import numpy as np
from classes.TreeGB import TreeGradientBoosting


class RandomForest:

    def __init__(self, max_features=-1, min_samples_split=2, max_depth=5, n_estimators=20, regression=True):
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.max_features = max_features
        self.max_depth = max_depth
        self.n_estimators = n_estimators
        self.trees = []
        self.regression=regression
        
    def fit(self, X, y):
        n_sample, n_feats = X.shape
        self.max_features = int(np.round(np.sqrt(n_feats))) if self.max_features == -1 else self.max_features

        for _ in range(0, self.n_estimators):
            # Boostraping to choose sample
            idx_sample = np.random.choice(range(n_sample), n_sample)

            # build trees
            clf = TreeGradientBoosting(max_depth=self.max_depth, min_samples_split=self.min_samples_split, tree_alone=True)
            clf.fit(X[idx_sample, :], y[idx_sample])
            self.trees.append(clf)

    def predict(self, X):
        # make a prediction with each tree
        predictions = np.array([tree.predict(X) for tree in self.trees])
        if not self.regression:
            # select the value the most common in each col of the array
            prediction_from_trees = [np.bincount(predictions[:, idx]).argmax() for idx in range(len(X))]
        else:
            prediction_from_trees = np.mean(predictions, axis=0)
        return prediction_from_trees
