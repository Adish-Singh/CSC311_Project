"""
This python file builds and trains our models:
- Logistic regression
- Random forest
- MLP

It then tests them, and saves the model parameters to a different file for pred.py to use.
"""
import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import log_loss
from sklearn.model_selection import GroupShuffleSplit
from sklearn.neural_network import MLPClassifier

# Defining important information from our data exploration
SEED = 311
TEST_IDS = [6, 13, 15, 16, 17, 18, 22, 31, 35, 38, 42, 49, 58, 69, 77, 82, 84, 87, 90, 94,
            103, 113, 119, 141, 152, 158, 162, 168, 178, 180, 190, 200, 206, 218, 258, 261,
            262, 266, 276, 279, 281, 282, 283, 287]

# Defining how data is split (No of folds)
K_FOLDS = 5

# Defining hyperparameter values for min token count
M_VALUES = [2, 3, 5, 10]

# Defining a tie-breaker, the simpler family wins
FAMILY_RANK = {"lr": 0, "rf": 1, "mlp": 2}

# ----------------------------------------------------------------------------
# Defining metrics used to evaluate our models
# ----------------------------------------------------------------------------


def accuracy(y, p):
    """ Defining the accuracy metric"""
    return float(np.mean(y == p))


def recalls(y, p, K):
    """Defining the recalls metric to be used in macro-averaged f1"""
    return np.array([np.mean(p[y == k] == k) if np.any(y == k) else np.nan for k in range(K)])


def macro_f1(y, p, K):
    """Defining the macro-averaged f1 metric"""
    f = []
    for k in range(K):
        tp = np.sum((p == k) & (y == k))
        prec = tp / max(np.sum(p == k), 1)
        rec = tp / max(np.sum(y == k), 1)
        f.append(0.0 if prec + rec == 0 else 2 * prec * rec / (prec + rec))
    return float(np.mean(f))


def confusion(y, p, K):
    """Creating the confusion matrix"""
    C = np.zeros((K, K), int)
    for a, b in zip(y, p):
        C[a, b] += 1
    return C


# --------------------------------------------------------------------------
# Creating the models
# --------------------------------------------------------------------------


def fit_model(family, p, X, y, ids):
    """Function to fit each type of model to the data
    -family = Model Family ("lr", "rf", "mlp")
    -p = Hyperparameters for model
    -X = Feature matrix
    -y = Target labels
    -ids = Student ID
    """
    if family == "lr":
        # solver = 'lbfgs' and penalty = 'l2' by default
        return LogisticRegression(C=p["C"], max_iter=5000, tol=1e-4, random_state=SEED).fit(X, y)

    if family == "rf":
        # criterion = 'gini' by default, n_jobs= -1 means using all processors
        return RandomForestClassifier(n_estimators=p["n_estimators"], max_depth=p["max_depth"],
                                      min_samples_leaf=p["min_leaf"], max_features=p["max_features"],
                                      random_state=SEED, n_jobs=-1).fit(X, y)

    if family == "mlp":
        # Can't simply use MLPClassifier from sklearn since we have to implement early stopping
        return fit_mlp(X, y, ids, p)

    raise ValueError(family)        # If incorrect family


def fit_mlp(X, y, ids, p, seed=SEED):
    """Helper function to create the MLP given the early stopping criteria"""
    # Hold out 10% of students for validation
    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=0.10,
        random_state=seed
    )

    # Generate the indices for training and testing
    train_idx, val_idx = next(splitter.split(X, y, groups=ids))

    # Defining Training data and labels and Validation data and labels
    Xt, yt = X[train_idx], y[train_idx]
    Xv, yv = X[val_idx], y[val_idx]

    # 3 possible classes
    classes = np.arange(3)

    # Initialize the MLP
    # activation = 'relu', solver = 'adam' , learning_rate = 'constant' by default
    # beta_1 = 0.9, beta_2 = 0.999, shuffle = True by default
    # max_iter = 1 since will manually train 1 epoch at a time
    clf = MLPClassifier(hidden_layer_sizes=p["hidden"], alpha=p["alpha"], batch_size=32, learning_rate_init=p["lr"],
                        max_iter=1, random_state=seed)

    # Initialize variables to store the data from the best epoch
    # Validation loss set to infinity for initial comparison
    best_loss = float("inf")
    best_weights = None
    best_intercepts = None

    # Count how many consecutive epochs have passed without improvement
    epochs_without_improvement = 0

    # Train epoch by epoch
    # Max 500 epochs
    for epoch in range(500):
        # On first call, give sklearn the possible output classes
        if epoch == 0:
            clf.partial_fit(Xt, yt, classes=classes)
        else:
            clf.partial_fit(Xt, yt)

        # Calculate validation cross-entropy loss
        probabilities = clf.predict_proba(Xv)           # Predict probabilities
        val_loss = log_loss(yv, probabilities, labels=classes)

        # Save weights if validation loss improves
        if val_loss < best_loss:
            best_loss = val_loss
            best_weights = [w.copy() for w in clf.coefs_]
            best_intercepts = [b.copy() for b in clf.intercepts_]
            epochs_without_improvement = 0

        else:
            epochs_without_improvement += 1

        # Stop if validation loss hasn't improved for 20 consecutive epochs
        if epochs_without_improvement >= 20:
            break

    # Restore weights from best epoch
    clf.coefs_ = best_weights
    clf.intercepts_ = best_intercepts

    return clf


def complexity(family, p):
    """Returns the models 'complexity' to be used to find simpler model"""
    if family == "lr":
        return p["C"], - p["m"]
    if family == "rf":
        return p["n_estimators"], p["max_depth"] or 10 ** 6, -p["min_leaf"], -p["m"]
    return sum(p["hidden"]), len(p["hidden"]), -p["m"]


# --------------------------------------------------------------------------
# Selecting the model
# --------------------------------------------------------------------------


