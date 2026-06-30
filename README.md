# Impact of Feature Scaling on Decision Trees

This repository contains a short, focused experiment demonstrating the effects of feature scaling on a `DecisionTreeClassifier` using `scikit-learn`. 

##  Project Overview
A common practice in preprocessing machine learning data is applying feature scaling (e.g., `StandardScaler`). While essential for distance-based algorithms (like KNN or SVM) and gradient-descent models (like Logistic Regression), it behaves differently with tree-based models. 

This notebook sets up a side-by-side comparison to test model performance on a dataset before and after standardization.

##  Key Findings
* **Model Used:** `DecisionTreeClassifier`
* **Accuracy (Unscaled Data):** `0.875`
* **Accuracy (Scaled Data):** `0.866`

### Core Insight
Decision Trees split nodes based on thresholds within a single feature at a time. Because of this, they are **invariant to monotonic transformations** like feature scaling. The minor discrepancy in performance (0.875 vs. 0.866) is entirely due to how decimal floating-point splits are calculated under the hood compared to discrete integer values, rather than any structural changes in how the tree learns.

##  Tech Stack & Libraries
* **Language:** Python 3
* **Libraries:** `pandas`, `scikit-learn`, `matplotlib`,numpy

##  Key Insights

1. **Tree Invariance to Scaling:** 
   Decision Trees split nodes based on threshold values within individual features (e.g., `Age > 35`). Because each feature is evaluated independently, scaling or shifting the data uniformly doesn't change the relative order of the values or the optimal split points. This makes tree-based models completely invariant to monotonic transformations like `StandardScaler`.

2. **Why the Slight Metric Difference?**
   You might notice a tiny variance in accuracy between the raw data (`0.875`) and the scaled data (`0.866`). This is not a structural change in the tree's behavior; rather, it is a known artifact of how floating-point arithmetic operates under the hood in `scikit-learn` when evaluating splits on scaled floating-point values versus original discrete integers.

3. **When to Actually Scale:**
   While scaling wasn't necessary for this Decision Tree, it remains crucial if you plan to compare this model against distance-based algorithms (like KNN or SVM) or gradient-descent-based models (like Logistic Regression) on the same dataset.
