# Decision Tree Algorithm: Built from Scratch

## Introduction

A decision tree is an algorithm for supervised learning that is not based on specific parameters. It is used for both classification and regression tasks. The tree has a hierarchical structure resembling a tree, with a starting point called the root node, branches, internal nodes, and endpoints known as leaf nodes.

The objective of the decision tree is to find the optimal splits for each node to minimise error and maximised the performance of the model. To find the optimal splits, splitting criteria such as the 'gini index' or 'entropy' has been utilised to act as a splitting guide for the tree to determine the best splitting possibility which can minimised the error and maximized the performance of the model. Additionally, hyperparameters such as max_depth, min_samples_split, and min_sample_leaf are enforced to prevent the overfitting of the decision tree.

This study has been based on the pseudocode provided by Hambali, M. A., Saheed, Y. K., Oladele, T. O., & Gbolagade, M. D. (2019)'s journal. The following flowchart of the Classification Tree can be seen below to make the algorithm easier to be understood:

![image](https://github.com/JordanCahya/decision_tree_algorithm/assets/115296804/ddc37ff2-9568-4662-aa8d-28beb01a3582)

A brief explanation of the flowchart is as follows:

- Data fitted to the decision tree algorithm

- The decision tree will grow according to the data fitted in the previous step by utilising the Gini index or entropy as the splitting criterion. The tree will grow recursively until the hyperparameter has been met or all or most of the data belonging to each of the partitions have the same class label (overfit stage).

- The tree will grow a leaf node and determine the output prediction of that node using a majority vote.

- Once, the new data or future data has been inputted, the future data will then travel across the tree to find the suitable nodes and find its prediction.

Please refer to the full article uploaded in https://medium.com/p/2f68c6ef0d5e

## Limitation and Recommendation

- Due to limited time and limited resources (the journal only provides detail on the classification tree), the codes produced are only suitable for classification datasets.

- The recommendation for future study is that the generic Decision Tree pseudocode can also be used for a regression tree. Should there be more time, this study could also develop a regression tree.

## References

- Hambali, M. A., Saheed, Y. K., Oladele, T. O., & Gbolagade, M. D. (2019). ADABOOST Ensemble Algorithms for Breast Cancer Classification. Journal of Advances in Computer Research, 10(2), 31–52. Retrieved from www.jacr.iausari.ac.ir

- PACMANN Academy. (2023). Jakarta

- Medium link: https://medium.com/p/2f68c6ef0d5e
