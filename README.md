# Decision Tree Algorithm: Built from Scratch

A decision tree is an algorithm for supervised learning that is not based on specific parameters. It is used for both classification and regression tasks. The tree has a hierarchical structure resembling a tree, with a starting point called the root node, branches, internal nodes, and endpoints known as leaf nodes.

The objective of the decision tree is to find the optimal splits for each node to minimise error and maximised the performance of the model. To find the optimal splits, splitting criteria such as the 'gini index' or 'entropy' has been utilised to act as a splitting guide for the tree to determine the best splitting possibility which can minimised the error and maximized the performance of the model. Additionally, hyperparameters such as max_depth, min_samples_split, and min_sample_leaf are enforced to prevent the overfitting of the decision tree.

This study has been based on the pseudocode provided by Hambali, M. A., Saheed, Y. K., Oladele, T. O., & Gbolagade, M. D. (2019)'s journal. However, with several modifications by adding hyperparameters, fit and predict functions, and other supporting functions. Therefore, a new pseudocode has been created which can be seen as follows.

![image](https://github.com/JordanCahya/decision_tree_algorithm/assets/115296804/0883717f-10c0-4c6c-a0ea-b5d5a3e08a56)

In addition, to make the understanding easire, the following flowchart of Classification Tree has been created:

![image](https://github.com/JordanCahya/decision_tree_algorithm/assets/115296804/ddc37ff2-9568-4662-aa8d-28beb01a3582)

A brief explanation of the flowchart is as follows:

- Data fitted to the decision tree algorithm

- The decision tree will grow according to the data fitted in the previous step by utilising the Gini index or entropy as the splitting criterion. The tree will grow recursively until the hyperparameter has been met or all or most of the data belonging to each of the partitions have the same class label (overfit stage).

- The tree will grow a leaf node and determine the output prediction of that node using a majority vote.

- Once, the new data or future data has been inputted, the future data will then travel across the tree to find the suitable nodes and find its prediction.
