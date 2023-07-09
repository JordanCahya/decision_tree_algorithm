import numpy as np
import pandas as pd
from ml_from_scratch.tree._criterion_class import _gini_index, _entropy, _mean_absolute_error, _mean_squared_error

class BaseDecisionTree:
    def __init__(self, max_depth, min_samples_split, min_sample_leaf):
        """
        Initialize the BaseDecisionTree.

        Parameters:
        - max_depth (int or None): The maximum depth of the decision tree. If None, the tree grows until all leaves are pure.
        - min_samples_split (int): The minimum number of samples required to perform a split.
        - min_sample_leaf (int): The minimum number of samples required to be at a leaf node.
        """
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_sample_leaf = min_sample_leaf
        self.tree = None

    def _fit(self, X, y):
        """
        Build the decision tree by fitting it to the given training data.

        Parameters:
        - X (array-like, shape (n_samples, n_features)): The input feature matrix.
        - y (array-like, shape (n_samples,)): The target values.

        Returns:
        None
        """
        if isinstance(X, pd.DataFrame):
            X = X.values
        else:
            X = X

        if isinstance(y, pd.Series):
            y = y.values
        else:
            y = y

        features = list(range(X.shape[1]))
        self.tree = self._grow_tree(X, y, features, self.max_depth)

    def _grow_tree(self, X, y, features, depth):
        """
        Recursively grow the decision tree by selecting the best splits.

        Parameters:
        - X (array-like, shape (n_samples, n_features)): The input feature matrix.
        - y (array-like, shape (n_samples,)): The target values.
        - features (list): The available feature indices.
        - depth (int): The current depth of the tree.

        Returns:
        - node (dict): The constructed tree node.
        """
        if self._stopping_condition(X, y, features, depth): # stopping condition is met (returning True)
            return self._create_node(y) # Create a leaf node which consider to be "pure" based on majority vote

        best_split, split_left, split_right, threshold = self._find_best_split(X, y, features)
        if best_split is None:
            return self._create_node(y)

        node = {'split_feature': best_split, 'children': {}}
        unique_values = set(X[:, best_split])

        for value in unique_values:
            if value <= threshold:
                child_X = X[split_left]
                child_y = y[split_left]
            else:
                child_X = X[split_right]
                child_y = y[split_right]

            if len(child_X) == 0:
                child_node = self._create_node(y)
            else:
                remaining_features = [f for f in features if f != best_split]
                child_node = self._grow_tree(child_X, child_y, remaining_features, depth - 1)

            node['children'][value] = child_node

        if 'default' not in node['children']:
            node['children']['default'] = self._create_node(y)

        return node

    def _stopping_condition(self, X, y, features, depth):
        """
        Check if the stopping condition for tree growth is met.

        Parameters:
        - X (array-like, shape (n_samples, n_features)): The input feature matrix.
        - y (array-like, shape (n_samples,)): The target values.
        - features (list): The available feature indices.
        - depth (int): The current depth of the tree.

        Returns:
        - stop (bool): True if the stopping condition is met, False otherwise.
        """
        if len(features) == 0: # If the features has been all used up for splitting then stop
            return True
        if len(set(y)) == 1: # If the output is equal to one then stop
            return True
        if depth >= self.max_depth: # if the depth of the tree is equal or more than max depth then stop
            return True
        if len(X) < self.min_samples_split or len(X) < self.min_sample_leaf:
            return True
        return False
 
    def _create_node(self, y):
        """
        Create a tree node with the label calculated from the target values.

        Parameters:
        - y (array-like, shape (n_samples,)): The target values.

        Returns:
        - node (dict): The tree node containing the label.
        """
        node = {'label': self._calculate_label(y)} # Create a leaf node returning final decision by utilising the majority vote
        return node

    def _calculate_label(self, y):
        """
        Calculate the label for a node based on the target values.

        Parameters:
        - y (array-like, shape (n_samples,)): The target values.

        Returns:
        - label: The calculated label for the node.
        """
        pass # The _calculate_label is implemented to the inherited tree where it is tailored to which decision tree utilised

    def _predict(self, X):
        """
        Predict the target values for the given input features.

        Parameters:
        - X (array-like, shape (n_samples, n_features)): The input feature matrix.

        Returns:
        - y_pred (array-like, shape (n_samples,)): The predicted target values.
        """
        if isinstance(X, pd.DataFrame):
            X = X.values

        y_pred = []
        for sample in X:
            y_pred.append(self._traverse_tree(sample, self.tree))

        return y_pred

    def _score(self, X, y):
        """
        Calculate the accuracy or mean squared error (MSE) / mean absolute error (MAE) of the tree on the given data.

        Parameters:
        - X (array-like, shape (n_samples, n_features)): The input feature matrix.
        - y (array-like, shape (n_samples,)): The true target values.

        Returns:
        - score (float): The accuracy / f1-score or MSE / MAE of the tree.
        """

        if isinstance(self, ClassificationTree):
            y_pred = self._predict(X)
            accuracy = np.mean(y_pred == y)
            score = print(f"Accuracy: {accuracy}") # Return accuracy and F1 score
        else:
            raise ValueError("Unknown tree type")
        return score

    def _traverse_tree(self, sample, node):
        """
        Traverse the decision tree recursively to predict the target value for a given sample.

        Parameters:
        - sample (array-like, shape (n_features,)): The input sample.
        - node (dict): The current node in the decision tree.

        Returns:
        - prediction: The predicted target value.
        """
        if 'label' in node:
            return node['label']

        split_feature = node['split_feature']
        value = sample[split_feature]
        if value in node['children']:
            child_node = node['children'][value]
        else:
            child_node = node['children']['default']

        return self._traverse_tree(sample, child_node)
    
    def _find_best_split(self, X, y, features):
        """
        Find the best feature to split on based on the chosen criterion.

        Parameters:
        - X (array-like, shape (n_samples, n_features)): The input feature matrix.
        - y (array-like, shape (n_samples,)): The target values.
        - features (list): The available feature indices.

        Returns:
        - best_split (int or None): The index of the best feature to split on, or None if no suitable split is found.
        - split_left (array-like, shape (n_samples,)): Boolean mask indicating the left subset of data.
        - split_right (array-like, shape (n_samples,)): Boolean mask indicating the right subset of data.
        - threshold (float or None): The threshold value used for splitting, or None if no suitable split is found.
        """
        best_split = None
        best_criterion_value = float('inf')
        split_left = None
        split_right = None
        threshold = None

        for feature in features:
            X_feature = X[:, feature]
            unique_values = np.unique(X_feature)

            for i in range(len(unique_values) - 1):
                current_threshold = (unique_values[i] + unique_values[i+1])/2.0
                current_split_left = X_feature <= current_threshold
                current_split_right = X_feature > current_threshold

                if self.criterion == 'gini':
                    criterion_value = _gini_index(X_feature, y)
                elif self.criterion == 'entropy':
                    criterion_value = _entropy(X_feature, y)
                else:
                    raise ValueError(f"Invalid criterion: {self.criterion}")

                if criterion_value < best_criterion_value:
                    best_criterion_value = criterion_value
                    best_split = feature
                    split_left = current_split_left
                    split_right = current_split_right
                    threshold = current_threshold

        return best_split, split_left, split_right, threshold

# ------------------------------
# ClassificationTree
# ------------------------------

class ClassificationTree(BaseDecisionTree):
    def __init__(self, criterion, max_depth, min_samples_split, min_sample_leaf):
        """
        Initialize the ClassificationTree.

        Parameters:
        - criterion (str): The criterion to measure the quality of a split. Can be 'gini' or 'entropy'.
        - max_depth (int or None): The maximum depth of the decision tree. If None, the tree grows until all leaves are pure.
        - min_samples_split (int): The minimum number of samples required to perform a split.
        - min_sample_leaf (int): The minimum number of samples required to be at a leaf node.
        """
        super().__init__(max_depth=max_depth, min_samples_split=min_samples_split, min_sample_leaf=min_sample_leaf)
        self.criterion = criterion

    def _calculate_label(self, y):
        """
        Calculate the label for a node based on the target values.

        Parameters:
        - y (array-like, shape (n_samples,)): The target values.

        Returns:
        - label: The calculated label for the node.
        """
        y_list = y.tolist()
        return max(set(y_list), key=y_list.count)

    def _fit(self, X, y):
        super()._fit(X, y)

    def _predict(self, X):
        return super()._predict(X)
    
    def _score(self, X, y):
        """
        Calculate the accuracy of the classification tree on the given data.

        Parameters:
        - X (array-like, shape (n_samples, n_features)): The input feature matrix.
        - y (array-like, shape (n_samples,)): The true target values.

        Returns:
        - accuracy (float): The accuracy of the classification tree.
        """
        return super()._score(X, y)