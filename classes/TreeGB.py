import numpy as np
from classes.TreeDecision import Tree, Node

class TreeGradientBoosting(Tree):
    def __init__(
        self,
        min_samples_split=2,
        max_depth=100,
        max_features=None,
        regression=True,
        tree_alone=False
    ):
        super().__init__(
            min_samples_split=min_samples_split,
            max_depth=max_depth,
            max_features=max_features,
            regression=regression,
        )
        self.tree_alone = tree_alone


    def fit(self, X, y, **kwargs):
        n_cols = X.shape[1]
        self.max_features = (
            X.shape[1] if not self.max_features else min(n_cols, self.max_features)
        )
        if not self.tree_alone:
            # self.update_previous_proba
            previous_proba = kwargs["previous_proba"]
        else:
            previous_proba = np.full_like(y, fill_value=np.mean(y))
        self.root = self._expend_tree(X, y, previous_proba=previous_proba)

    def _expend_tree(self, X, y, previous_proba, depth=0):
        n_samples, n_feats = X.shape
        num_label = len(np.unique(y))

        # Stopping crietria
        if (
            depth > self.max_depth
            or num_label == 1
            or n_samples <= self.min_samples_split
        ):
            return Node(
                value=np.mean(y)
            )

        # select randomly max_features
        features = np.random.choice(n_feats, self.max_features, replace=False)

        # splitting the data with the feature which max the information gain
        feat_split, feat_thr, right_index_opt, left_index_opt = self._optimal_split(
            features, X, y
        )

        # Create the child nodes
        # give to each node the data splited with the optimal threshold
        right_child = self._expend_tree(
            X[right_index_opt, :],
            y[right_index_opt],
            previous_proba[right_index_opt],
            depth + 1,
        )
        left_child = self._expend_tree(
            X[left_index_opt, :],
            y[left_index_opt],
            previous_proba[left_index_opt],
            depth + 1,
        )
        return Node(feat_split, feat_thr, left_child, right_child)

    def predict(self, X):
        return [self._travelling_tree(val_row, self.root) for val_row in X]

    def _travelling_tree(self, values_row, node):

        # criteria condition: attribute value when we reach the last leaf
        if node.value is not None:
            return node.value

        if values_row[node.feature] <= node.threshold:
            return self._travelling_tree(values_row, node.left)
        return self._travelling_tree(values_row, node.right)