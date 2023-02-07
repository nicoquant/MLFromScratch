import numpy as np
from collections import Counter
import cProfile


class Node:
    def __init__(self, feature=None, threshold=None, left=None, right=None, value=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value


class Tree:
    def __init__(self, min_samples_split=2, max_depth=100, max_features=None, regression=False):
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.max_features = max_features
        self.root = None
        self.regression = regression

    def fit(self, X, y):
        n_cols = X.shape[1]
        self.max_features = X.shape[1] if not self.max_features else min(n_cols, self.max_features)
        self.root = self._expend_tree(X, y)

    def _expend_tree(self, X, y, depth=0):

        n_samples, n_feats = X.shape
        num_label = len(np.unique(y))

        # Stopping crietria
        if depth >= self.max_depth or num_label == 1 or n_samples < self.min_samples_split:
            # leaf_val is the value of the leaf
            leaf_val = sum(y > 0) > sum(y <= 0) if not self.regression else np.mean(y)
            return Node(value=leaf_val)

        # select randomly max_features
        if self.max_features == X.shape[1]:
            features = np.random.choice(n_feats, self.max_features, replace=False)
        else:
            # define the index for the features
            features = range(self.max_features+1)

        # splitting the data with the feature which max the information gain
        feat_split, feat_thr, right_index_opt, left_index_opt = self._optimal_split(features, X, y)

        # Create the child nodes
        # give to each node the data splited with the optimal threshold
        right_child = self._expend_tree(X[right_index_opt, :], y[right_index_opt], depth + 1)
        left_child = self._expend_tree(X[left_index_opt, :], y[left_index_opt], depth + 1)
        return Node(feat_split, feat_thr, left_child, right_child)


    def _optimal_split(self, features, X, y):
        # start gain at -inf
        splitting_eval_opt = -np.inf

        # initialize variable for PEP8
        feat_opt = None
        thr_opt = None
        right_index_opt = None
        left_index_opt = None
        # test each feature
        for each in features:
            X_col = X[:, each]
            threshold = np.unique(X_col)
            # go through each value in the feature to define the best one to split on
            for thr in threshold:
                if not self.regression:
                    splitting_eval, right_index, left_index = self._information_gain(X_col, y, thr)
                else:
                    splitting_eval, right_index, left_index = self._mse(X_col, y, thr)

                # switch in negative the mse because you have to minimize it
                # yes you have to max information gain

                if splitting_eval > splitting_eval_opt:
                    # Select the parameters optimal for the split
                    splitting_eval_opt = splitting_eval
                    right_index_opt = right_index
                    left_index_opt = left_index
                    feat_opt = each
                    thr_opt = thr

        return feat_opt, thr_opt, right_index_opt, left_index_opt

    def _information_gain(self, X_col, y, thr):
        length = len(y)
        # entropy state
        entropy_parent = self._entropy(y)

        # define feature data higher or lower than the threshold
        child_right_index = np.where(X_col > thr)[0].flatten()
        child_left_index = np.where(X_col <= thr)[0].flatten()

        if len(child_right_index) == 0 or len(child_left_index) == 0:
            return 0, child_right_index, child_left_index

        # compute entropy of legs right and left
        entropy_right, entropy_left = self._entropy(y[child_right_index]), self._entropy(y[child_left_index])

        # define weights for entropy of child
        weights = np.divide([len(child_right_index), len(child_left_index)], length)
        entropy_child = np.sum(entropy_right * weights[0] + entropy_left * weights[1])

        return entropy_parent - entropy_child, child_right_index, child_left_index

    @staticmethod
    def _entropy(y):
        length = len(y)
        c = Counter(y)
        count = np.array([c.most_common()[i][1] for i in range(len(c.most_common()))])#np.bincount(y)
        all_p = np.divide(count, length)
        return -np.sum([p * np.log(p) for p in all_p])

    def _mse(self, X_col, y, thr):
        child_right_index = np.where(X_col > thr)[0].flatten()
        child_left_index = np.where(X_col <= thr)[0].flatten()

        if len(child_right_index) == 0 or len(child_left_index) == 0:
            return -np.inf, child_right_index, child_left_index

        # compute mse of each leg
        mse_right, mse_left = self._mse_calcul(y[child_right_index]), self._mse_calcul(y[child_left_index])
        RSS = mse_right + mse_left
        # switch in negative the rss because you have to minimize it and following this we maximum the criteria
        return -RSS, child_right_index, child_left_index


    @staticmethod
    def _mse_calcul(y_pred):
        return np.sum((y_pred - np.mean(y_pred))**2)

    def predict(self, X):
        return [self._travelling_tree(val_row, self.root) for val_row in X]

    def _travelling_tree(self, values_row, node):

        # criteria condition: attribute value when we reach the last leaf
        if node.value is not None:
            return node.value

        if values_row[node.feature] <= node.threshold:
            return self._travelling_tree(values_row, node.left)
        return self._travelling_tree(values_row, node.right)


