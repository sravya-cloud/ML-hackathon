"""Model zoo for PathSnap candidate classification.

Spans multiple algorithm families (the hackathon requires >=5 models before
selecting a final one):

  linear/probabilistic : Logistic Regression, Gaussian Naive Bayes
  instance-based       : K-Nearest Neighbours
  tree-based           : Decision Tree
  ensemble (bagging)   : Random Forest
  boosting             : HistGradientBoosting, XGBoost
  kernel-based         : RBF SVM
  neural network       : MLP

Scale-sensitive models (LogReg, NB, KNN, SVM, MLP) are wrapped with a
StandardScaler. Some O(n^2)-ish models (SVM, KNN) are trained on a bounded
subsample for tractability on ~1.2M rows; this is flagged in `subsample`.
"""
from __future__ import annotations

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier

try:
    from xgboost import XGBClassifier
    _HAS_XGB = True
except Exception:
    _HAS_XGB = False


def _scaled(est):
    return Pipeline([("scaler", StandardScaler()), ("clf", est)])


def model_zoo(seed=42):
    """Return {name: {estimator, family, config, subsample}}."""
    zoo = {}

    zoo["LogisticRegression"] = dict(
        estimator=_scaled(LogisticRegression(max_iter=2000, C=1.0,
                                             class_weight="balanced", random_state=seed)),
        family="Linear / probabilistic",
        config="C=1.0, class_weight=balanced, StandardScaler",
        subsample=None)

    zoo["GaussianNB"] = dict(
        estimator=_scaled(GaussianNB()),
        family="Probabilistic", config="default, StandardScaler", subsample=None)

    zoo["KNN"] = dict(
        estimator=_scaled(KNeighborsClassifier(n_neighbors=25, weights="distance",
                                               n_jobs=-1)),
        family="Instance-based", config="k=25, distance-weighted, StandardScaler",
        subsample=120_000)

    zoo["DecisionTree"] = dict(
        estimator=DecisionTreeClassifier(max_depth=12, min_samples_leaf=50,
                                         class_weight="balanced", random_state=seed),
        family="Tree-based", config="max_depth=12, min_samples_leaf=50", subsample=None)

    zoo["RandomForest"] = dict(
        estimator=RandomForestClassifier(n_estimators=150, max_depth=18,
                                         min_samples_leaf=20, n_jobs=-1,
                                         class_weight="balanced_subsample",
                                         random_state=seed),
        family="Ensemble (bagging)",
        config="150 trees, max_depth=18, min_samples_leaf=20", subsample=None)

    zoo["HistGradientBoosting"] = dict(
        estimator=HistGradientBoostingClassifier(max_iter=400, learning_rate=0.08,
                                                 max_depth=None, l2_regularization=1.0,
                                                 random_state=seed),
        family="Boosting", config="400 iters, lr=0.08, l2=1.0", subsample=None)

    if _HAS_XGB:
        zoo["XGBoost"] = dict(
            estimator=XGBClassifier(n_estimators=400, max_depth=7, learning_rate=0.08,
                                    subsample=0.9, colsample_bytree=0.9,
                                    tree_method="hist", eval_metric="logloss",
                                    n_jobs=-1, random_state=seed),
            family="Boosting", config="400 trees, depth=7, lr=0.08, hist", subsample=None)

    zoo["RBF_SVM"] = dict(
        estimator=_scaled(SVC(C=2.0, gamma="scale", probability=True,
                              class_weight="balanced", random_state=seed)),
        family="Kernel-based", config="RBF, C=2.0, gamma=scale, StandardScaler",
        subsample=40_000)

    zoo["MLP"] = dict(
        estimator=_scaled(MLPClassifier(hidden_layer_sizes=(64, 32), alpha=1e-4,
                                        max_iter=60, early_stopping=True,
                                        random_state=seed)),
        family="Neural network", config="(64,32), ReLU, Adam, early stopping",
        subsample=None)

    return zoo
