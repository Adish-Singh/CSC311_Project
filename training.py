"""
This python file builds and trains our models:
- Logistic regression
- Random forest
- MLP

It then tests them, and saves everything pred.py needs (preprocessing state and
model parameters) into ONE json file, model.json, in the same folder as this code.
"""
import argparse
import json
import os

import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import log_loss
from sklearn.model_selection import GroupShuffleSplit
from sklearn.neural_network import MLPClassifier

import pred
from pred import Preprocessor, ID_COL, LABEL_COL

# Folder containing this file (model.json is saved here by default)
HERE = os.path.dirname(os.path.abspath(__file__))

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
# Splitting our data (same function as in the data exploration)
# ----------------------------------------------------------------------------


def split_by_student(df, n_test=44, seed=SEED):
    """Splits the unique student ids into train/test groups. All three rows
    from the same student end up in the same split. Returns two boolean
    arrays (is_train, is_test), one entry per row."""
    rng = np.random.default_rng(seed)
    test_ids = rng.choice(df[ID_COL].unique(), n_test, replace=False)
    is_test = df[ID_COL].isin(test_ids).to_numpy()
    return ~is_test, is_test


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


def one_se_choice(results):
    """Applies the one-standard-error rule to a list of CV results.
    Finds the best mean accuracy, then returns the simplest setting whose mean
    is within one standard error (sd / sqrt(K_FOLDS)) of it."""
    best = max(results, key=lambda r: r["mean"])
    se = best["std"] / np.sqrt(K_FOLDS)

    # Every setting that is statistically indistinguishable from the best
    close_enough = [r for r in results if r["mean"] >= best["mean"] - se]

    # Among those, simplest first; higher accuracy breaks ties
    chosen = min(close_enough, key=lambda r: (complexity(r["family"], r["params"]), -r["mean"]))
    return chosen, best, se


def pick(rng, options):
    """Picks one element uniformly at random from a list (works with None / tuples)"""
    return options[int(rng.integers(len(options)))]


def candidates(family, n_random, rng, m_values):
    """Generates the hyperparameter settings to try for one model family.
    LR: full grid (C x m). RF / MLP: n_random random draws."""
    if family == "lr":
        return [{"C": float(10.0 ** e), "m": m} for e in range(-3, 3) for m in m_values]

    out = []
    for _ in range(n_random):
        if family == "rf":
            out.append({"n_estimators": pick(rng, [100, 200, 300, 500]),
                        "max_depth": pick(rng, [5, 10, 15, 20, None]),
                        "min_leaf": pick(rng, [1, 2, 4, 8]),
                        "max_features": pick(rng, ["sqrt", "log2", 0.2]),
                        "m": pick(rng, m_values)})
        else:   # mlp; alpha and learning rate are log-uniform
            out.append({"hidden": pick(rng, [(32,), (64,), (128,), (64, 32)]),
                        "alpha": float(10 ** rng.uniform(-5, -1)),
                        "lr": float(10 ** rng.uniform(-4, -2)),
                        "m": pick(rng, m_values)})
    return out


# --------------------------------------------------------------------------
# Cross-validation
# --------------------------------------------------------------------------


class FeatureCache:
    """Builds (and remembers) the feature matrices for each fold and each m.
    The Preprocessor is fit on the training part of the fold ONLY, so nothing
    from the validation students leaks into medians, means or vocabularies."""

    def __init__(self, df, folds):
        self.df, self.folds, self.cache = df, folds, {}

    def get(self, fold_i, m):
        if (fold_i, m) not in self.cache:
            tr, va = self.folds[fold_i]
            pre = Preprocessor(m).fit(self.df.iloc[tr])
            self.cache[(fold_i, m)] = (pre.transform(self.df.iloc[tr]),
                                       pre.transform(self.df.iloc[va]))
        return self.cache[(fold_i, m)]


def cv_eval(family, p, cache, folds, y, ids, K=3):
    """Runs one hyperparameter setting through all folds.
    Returns mean/sd of accuracy and macro-F1, mean per-class recall, mean
    training accuracy, and the out-of-fold predictions (for the confusion matrix)."""
    accs, f1s, train_accs, recs = [], [], [], []
    oof = np.zeros(len(y), int)

    for i, (tr, va) in enumerate(folds):
        Xtr, Xva = cache.get(i, p["m"])
        model = fit_model(family, p, Xtr, y[tr], ids[tr])
        pv = model.predict(Xva)
        oof[va] = pv
        accs.append(accuracy(y[va], pv))
        f1s.append(macro_f1(y[va], pv, K))
        recs.append(recalls(y[va], pv, K))
        train_accs.append(accuracy(y[tr], model.predict(Xtr)))

    accs = np.array(accs)
    return {"family": family, "params": p, "fold_acc": accs.tolist(),
            "mean": float(accs.mean()), "std": float(accs.std(ddof=1)),
            "f1": float(np.mean(f1s)), "f1_std": float(np.std(f1s, ddof=1)),
            "recall": np.mean(recs, axis=0).tolist(),
            "train_acc": float(np.mean(train_accs)),
            "oof": oof}


# --------------------------------------------------------------------------
# Saving the model for pred.py, and checking the NumPy version matches sklearn
# --------------------------------------------------------------------------


def export(pre, model, family, classes, outdir):
    """Writes ONE file, outdir/model.json, holding the fitted preprocessing state,
    the model parameters."""
    os.makedirs(outdir, exist_ok=True)

    if family == "lr":
        params = {"W": model.coef_.T.tolist(), "b": model.intercept_.tolist()}

    elif family == "mlp":
        params = {"n_layers": len(model.coefs_)}
        for i, (W, b) in enumerate(zip(model.coefs_, model.intercepts_)):
            params[f"W{i}"], params[f"b{i}"] = W.tolist(), b.tolist()

    else:   # rf: concatenate all trees into flat arrays
        trees = [e.tree_ for e in model.estimators_]
        offsets = np.concatenate([[0], np.cumsum([t.node_count for t in trees])[:-1]])
        left, right, feat, thr, val = [], [], [], [], []
        for t, o in zip(trees, offsets):
            leaf = t.children_left == -1
            me = np.arange(t.node_count) + o
            left.append(np.where(leaf, me, t.children_left + o))    # leaves point to themselves
            right.append(np.where(leaf, me, t.children_right + o))
            feat.append(np.where(leaf, 0, t.feature))
            thr.append(np.where(leaf, 0.0, t.threshold))
            v = t.value[:, 0, :]
            val.append(np.round(v / v.sum(1, keepdims=True), 6))    # class probabilities
        params = {"left": np.concatenate(left).tolist(), "right": np.concatenate(right).tolist(),
                  "feat": np.concatenate(feat).tolist(), "thr": np.concatenate(thr).tolist(),
                  "val": np.concatenate(val).tolist(), "roots": offsets.tolist(),
                  "max_depth": int(max(t.max_depth for t in trees) + 1)}

    data = {"kind": family, "classes": [str(c) for c in classes],
            "preproc": pre.get_state(), "model": params}

    path = os.path.join(outdir, "model.json")
    with open(path, "w") as f:
        json.dump(data, f, separators=(",", ":"))
    size = os.path.getsize(path)
    print(f"  saved {path} ({size / 1e6:.2f} MB; submission limit is 10 MB)")
    if size > 10e6:
        print("  WARNING: model.json is over 10 MB - use fewer trees / a shallower forest")


def verify(model, df, path):
    """Reloads model.json through pred.py and checks that its NumPy predictions
    equal sklearn's."""
    pre, M = pred.load_artifacts(path)
    X = pre.transform(df)
    P_numpy, P_sklearn = pred.predict_proba(X, M), model.predict_proba(X)
    err = float(np.abs(P_numpy - P_sklearn).max())
    agree = float(np.mean(P_numpy.argmax(1) == P_sklearn.argmax(1)))
    print(f"  verification: max |P_numpy - P_sklearn| = {err:.2e}, label agreement = {agree:.4f}")
    assert agree == 1.0 and err < 1e-3, "NumPy re-implementation does not match sklearn!"


def plot_hparam(all_results, out_path):
    """Validation (±1 sd) and training accuracy vs the key hyperparameter of each
    family, to check the range is sensible and to read off over/underfitting."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    key_for = {"lr": ("C", True), "rf": ("max_depth", False), "mlp": ("alpha", True)}
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, (fam, (key, log_x)) in zip(axes, key_for.items()):
        R = [r for r in all_results if r["family"] == fam]
        x = np.array([30 if r["params"][key] is None else r["params"][key] for r in R], float)
        ax.errorbar(x, [r["mean"] for r in R], yerr=[r["std"] for r in R],
                    fmt="o", alpha=0.5, label="validation (±1 sd)")
        ax.scatter(x, [r["train_acc"] for r in R], marker="x", c="r", alpha=0.5, label="train")
        if log_x:
            ax.set_xscale("log")
        ax.set_xlabel(key + (" (30 = none)" if key == "max_depth" else ""))
        ax.set_ylabel("accuracy")
        ax.set_title(fam.upper())
        ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"  saved {out_path}")


def check_features_ignore_id_and_label(df):
    """Safety check: the features must NOT depend on the student id or the label.
    ids are only used to group rows (splits, CV folds, early-stopping holdout);
    we scramble both columns and make sure the feature matrix is unchanged."""
    assert ID_COL not in pred.REQUIRED_COLS and LABEL_COL not in pred.REQUIRED_COLS
    pre = Preprocessor(3).fit(df)
    assert not any(n.lower().startswith(("id", "label")) for n in pre.names), "id/label leaked into features"
    scrambled = df.copy()
    rng = np.random.default_rng(0)
    scrambled[ID_COL] = rng.permutation(scrambled[ID_COL].values)
    scrambled[LABEL_COL] = rng.permutation(scrambled[LABEL_COL].values)
    same = np.array_equal(pre.transform(df), Preprocessor(3).fit(scrambled).transform(scrambled))
    assert same, "features changed when id/label were scrambled -> they are leaking into X"
    print("Check passed: features do not use the student id or the label.")


# --------------------------------------------------------------------------
# Main: the whole pipeline
# --------------------------------------------------------------------------


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.path.join(HERE, "csc311_project_training_202609.csv"),
                    help="training CSV (default: csc311_project_training_202609.csv next to this file)")
    ap.add_argument("--out", default=HERE, help="folder for model.json (default: same folder as train.py)")
    ap.add_argument("--quick", action="store_true", help="tiny search, for smoke-testing")
    ap.add_argument("--plots", action="store_true")
    ap.add_argument("--no-refit", action="store_true",
                    help="export the model trained on the 738 training rows instead of refitting on all rows")
    args = ap.parse_args()
    np.random.seed(SEED)

    # ---- 1. Load data and hold out the test students
    if not os.path.exists(args.data):
        raise SystemExit(f"Can't find the data file: {args.data}\n"
                         f"Put csc311_project_training_202609.csv in {HERE} or pass --data path/to/file.csv")
    df = pd.read_csv(args.data)
    id_col, label_col = ID_COL, LABEL_COL

    classes = sorted(df[label_col].unique())
    K = len(classes)
    y_all = df[label_col].map({c: i for i, c in enumerate(classes)}).values

    # Same split as the data exploration; check it matches the ids listed in the proposal
    is_train, is_test = split_by_student(df)
    if sorted(int(i) for i in df.loc[is_test, id_col].unique()) != TEST_IDS:
        print("WARNING: the test students differ from TEST_IDS (Appendix A of the proposal)!")
    print(f"Test: {df.loc[is_test, id_col].nunique()} students, {is_test.sum()} rows | "
          f"Train: {df.loc[~is_test, id_col].nunique()} students, {(~is_test).sum()} rows")
    train_df, test_df = df[~is_test].reset_index(drop=True), df[is_test].reset_index(drop=True)
    y_tr, y_te = y_all[~is_test], y_all[is_test]
    ids_tr, ids_te = train_df[id_col].values, test_df[id_col].values

    check_features_ignore_id_and_label(train_df)
    probe = Preprocessor(3).fit(train_df)
    n_text = sum(len(v) for v in probe.vocab.values())
    print(f"Features at m=3: {len(probe.names)} total, {n_text} text tokens "
          f"(the data exploration reported 465)")

    # 5 folds split by student: shuffle the student ids, cut into 5 chunks
    rng = np.random.default_rng(SEED)
    chunks = np.array_split(rng.permutation(np.unique(ids_tr)), K_FOLDS)
    folds = [(np.where(~np.isin(ids_tr, c))[0], np.where(np.isin(ids_tr, c))[0]) for c in chunks]

    # ---- 2. Hyperparameter search (same folds for every model)
    n_random = 6 if args.quick else 60
    m_values = [3, 5] if args.quick else M_VALUES
    cache = FeatureCache(train_df, folds)
    all_results, best_per_family = [], {}

    for fam in ["lr", "rf", "mlp"]:
        settings = candidates(fam, n_random, np.random.default_rng(SEED), m_values)
        print(f"\n[{fam.upper()}] {len(settings)} configurations")
        results = []
        for j, p in enumerate(settings):
            r = cv_eval(fam, p, cache, folds, y_tr, ids_tr, K)
            results.append(r)
            print(f"  {j + 1:3d}/{len(settings)} acc={r['mean']:.3f}±{r['std']:.3f} {p}")
        all_results += results

        chosen, best, se = one_se_choice(results)
        best_per_family[fam] = chosen
        print(f"  best {best['mean']:.3f} (SE {se:.3f}) -> 1-SE choice {chosen['mean']:.3f} {chosen['params']}")

        # Warn if the chosen value sits at the edge of its search range
        edge = {"lr": ("C", 1e-3, 1e2), "mlp": ("alpha", 1e-5, 1e-1)}.get(fam)
        if edge and chosen["params"][edge[0]] in edge[1:]:
            print(f"  WARNING: best {edge[0]} is at the edge of its range - consider extending it")

    finalists = dict(best_per_family)

    # ---- 3. Choose the final family: best mean, then 1-SE in favour of the simpler family
    top = max(finalists.values(), key=lambda r: r["mean"])
    se = top["std"] / np.sqrt(K_FOLDS)
    near_top = [r for r in finalists.values() if r["mean"] >= top["mean"] - se]
    final = min(near_top, key=lambda r: (FAMILY_RANK[r["family"]], -r["mean"]))
    fam, params = final["family"], final["params"]

    print("\n=== Model comparison (5-fold grouped CV, mean ± sd over folds) ===")
    print(f"{'family':8s}{'acc':>16s}{'macro-F1':>16s}{'train acc':>11s}  recalls")
    for name, r in finalists.items():
        tag = name
        rc = " ".join(f"{classes[k]}={v:.2f}" for k, v in enumerate(r["recall"]))
        print(f"{tag:8s}{r['mean']:9.3f}±{r['std']:.3f}{r['f1']:9.3f}±{r['f1_std']:.3f}"
              f"{r['train_acc']:11.3f}  {rc}")
    print(f"\nSELECTED: {fam} {params}")
    print("Pooled out-of-fold confusion matrix (rows = true):", classes)
    print(confusion(y_tr, final["oof"], K))

    if args.plots:
        os.makedirs(args.out, exist_ok=True)
        plot_hparam(all_results, os.path.join(args.out, "hparam_curves.png"))

    # ---- 4. Retrain on the 738 training rows; evaluate on the test set exactly once
    pre = Preprocessor(params["m"]).fit(train_df)
    Xtr, Xte = pre.transform(train_df), pre.transform(test_df)
    model = fit_model(fam, params, Xtr, y_tr, ids_tr)
    pt = model.predict(Xte)
    test_acc = accuracy(y_te, pt)

    # 95% CI: bootstrap over STUDENTS (a student's 3 rows stay together)
    brng = np.random.default_rng(SEED)
    students = np.unique(ids_te)
    rows_of = {s: np.where(ids_te == s)[0] for s in students}
    boots = []
    for _ in range(1000):
        rows = np.concatenate([rows_of[s] for s in brng.choice(students, len(students), replace=True)])
        boots.append(accuracy(y_te[rows], pt[rows]))
    lo, hi = np.percentile(boots, [2.5, 97.5])
    print(f"\nTEST accuracy = {test_acc:.3f}  (95% bootstrap CI over students: [{lo:.3f}, {hi:.3f}])")
    print(f"TEST macro-F1 = {macro_f1(y_te, pt, K):.3f}; recalls:",
          {classes[k]: round(float(v), 3) for k, v in enumerate(recalls(y_te, pt, K))})

    # ---- 5. Final model
    # The held-out students were only used to estimate performance (stage 4). The real test set is
    # the professor's, so by default we refit the chosen configuration on ALL rows and export that.
    if args.no_refit:
        print("\nExporting the model trained on the training rows only (--no-refit)...")
        export(pre, model, fam, classes, args.out)
        verify(model, test_df, args.out)
    else:
        print("\nRefitting the selected configuration on ALL rows and exporting...")
        pre_all = Preprocessor(params["m"]).fit(df)
        X_all = pre_all.transform(df)
        final_model = fit_model(fam, params, X_all, y_all, df[id_col].values)
        print(f"  trained on {len(df)} rows ({df[id_col].nunique()} students); "
              f"accuracy on these rows (training accuracy) = {accuracy(y_all, final_model.predict(X_all)):.3f}")
        export(pre_all, final_model, fam, classes, args.out)
        verify(final_model, df, args.out)

    with open(os.path.join(args.out, "cv_results.json"), "w") as f:
        json.dump([{k: v for k, v in r.items() if k != "oof"} for r in all_results],
                  f, default=str, indent=1)
    print("Done. Submit pred.py together with model.json.")


if __name__ == "__main__":
    main()
