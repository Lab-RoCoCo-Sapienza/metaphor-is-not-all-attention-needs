"""The kernel probe: support vector classifier with an RBF kernel."""
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC


def tuned(grid, cv, max_iter, seed):
    """Table 1: class-balanced, with C chosen on the inner folds by ROC-AUC."""
    return GridSearchCV(SVC(class_weight="balanced", max_iter=max_iter, random_state=seed),
                        {"C": grid}, cv=cv, scoring="roc_auc")


def fixed(C):
    """A.11: a fixed C."""
    return SVC(C=C)
