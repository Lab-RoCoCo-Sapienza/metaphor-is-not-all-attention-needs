"""The linear probe: logistic regression."""
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV


def tuned(grid, cv, max_iter, seed):
    """Table 1: L1-regularized, class-balanced, with C chosen on the inner folds by ROC-AUC."""
    return LogisticRegressionCV(Cs=grid, cv=cv, penalty="l1", solver="liblinear", scoring="roc_auc",
                                class_weight="balanced", max_iter=max_iter, random_state=seed, refit=True)


def fixed(C, max_iter, class_weight=None):
    """A.11 and A.7: L2-regularized at a fixed C."""
    return LogisticRegression(C=C, max_iter=max_iter, class_weight=class_weight)
