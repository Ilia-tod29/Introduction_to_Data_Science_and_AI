"""
Question 2: classification of the Iris data set

Run:    python assignment2_question2.py
Needs:  matplotlib, scikit-learn
Output: figure PNG files (saved, not shown) + results printed to the terminal
"""

import matplotlib
matplotlib.use("Agg")                     # save figures to files, no window needed
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, confusion_matrix
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.neighbors import KNeighborsClassifier

# ----------------------------------------------------------------------------
# Data: 150 flowers, 4 measurements (cm), 3 species (50 flowers each)
# 70% is used for training and 30% is held out for testing, so the models are
# judged on flowers they have not seen. stratify keeps the species balanced
# (35 of each in training, 15 of each in the test set); random_state fixes the
# split so the results are the same on every run.
# ----------------------------------------------------------------------------
iris = load_iris()
X_train, X_test, y_train, y_test = train_test_split(
    iris.data, iris.target, test_size=0.3, stratify=iris.target, random_state=42)

# A first look at the data: smallest - largest value of each measurement, per species
print(f"[DATA]  {len(iris.data)} flowers, measurements in cm (smallest - largest per species)")
print(f"{'':<12}{'flowers':>8}" + "".join(f"{name[:-5]:>16}" for name in iris.feature_names))
for i, name in enumerate(iris.target_names):
    flowers = iris.data[iris.target == i]
    print(f"{name:<12}{len(flowers):>8}" + "".join(
        f"{low:>10.1f} - {high:.1f}" for low, high in zip(flowers.min(axis=0), flowers.max(axis=0))))
print()

# ----------------------------------------------------------------------------
# 2a. Logistic regression, evaluated with a confusion matrix
# ----------------------------------------------------------------------------
# max_iter=1000 only gives the solver enough steps to converge
log_reg = LogisticRegression(max_iter=1000).fit(X_train, y_train)
y_pred = log_reg.predict(X_test)

# Rows = true species, columns = predicted species; the diagonal is correct
print("[2a  LOGISTIC REGRESSION]")
print("Confusion matrix (rows = true, columns = predicted):")
print(confusion_matrix(y_test, y_pred))
log_reg_acc = accuracy_score(y_test, y_pred)
print(f"Test accuracy: {log_reg_acc:.3f} "
      f"({(y_test == y_pred).sum()} of {len(y_test)} flowers correct)")

ConfusionMatrixDisplay.from_predictions(
    y_test, y_pred, display_labels=iris.target_names, cmap="Blues", colorbar=False)
plt.title("Figure 5: Confusion matrix, logistic regression (test set)")
plt.tight_layout()
plt.savefig("figure5_confusion_matrix_logistic_regression.png", dpi=200)
plt.close()

# ----------------------------------------------------------------------------
# 2b. k-nearest neighbours for every k, with uniform and distance weights
#     uniform:  each of the k nearest neighbours has one vote
#     distance: each vote counts 1 / distance, so closer neighbours count more
# ----------------------------------------------------------------------------
WEIGHTS = ["uniform", "distance"]
K_VALUES = range(1, len(X_train) + 1)     # 1 .. 105 (all training flowers)
SHOWN_K = [1, 5, 15, 50, 75, 105]         # values of k shown in the tables and in Figure 7

# accuracy[weights][k - 1] = (training accuracy, test accuracy)
accuracy = {w: [] for w in WEIGHTS}
for w in WEIGHTS:
    for k in K_VALUES:
        knn = KNeighborsClassifier(n_neighbors=k, weights=w).fit(X_train, y_train)
        accuracy[w].append((knn.score(X_train, y_train), knn.score(X_test, y_test)))

print("\n[2b  K-NEAREST NEIGHBOURS]  accuracy on the training and test data")
print(f"{'k':>4} {'Uniform train':>15} {'Uniform test':>14} {'Distance train':>16} {'Distance test':>15}")
for k in SHOWN_K:
    (u_train, u_test), (d_train, d_test) = (accuracy[w][k - 1] for w in WEIGHTS)
    print(f"{k:>4} {u_train:>15.1%} {u_test:>14.1%} {d_train:>16.1%} {d_test:>15.1%}")

# Test accuracy against k, with logistic regression as a reference line
fig, ax = plt.subplots(figsize=(8, 5.5))
for w in WEIGHTS:
    ax.plot(K_VALUES, [test for _, test in accuracy[w]], label=f"k-NN, {w} weights")
ax.axhline(log_reg_acc, color="gray", ls="--", label="Logistic regression")
ax.set(xlabel="Number of neighbours (k)", ylabel="Test accuracy",
       title="Figure 6: k-NN test accuracy for growing k")
ax.grid(alpha=0.3); ax.legend()
fig.tight_layout(); fig.savefig("figure6_knn_accuracy_vs_k.png", dpi=200); plt.close(fig)

# ----------------------------------------------------------------------------
# 2c. Compare logistic regression and k-NN: confusion matrices + accuracy
#     The test set is small (one flower = 2.2 percentage points), so each model
#     is also scored with 10-fold cross-validation: the data is cut into 10
#     parts, each part is used once as the test set, and the 10 accuracies are
#     averaged. Here the models train on 135 flowers (45 per species), not 105,
#     so k = 105 is no longer "all training flowers" in the CV column.
# ----------------------------------------------------------------------------
cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)


def evaluate(name, model):
    """Print one row of the comparison table and return the test predictions."""
    y_pred = model.fit(X_train, y_train).predict(X_test)
    cv_acc = cross_val_score(model, iris.data, iris.target, cv=cv).mean()
    print(f"{name:<22} {accuracy_score(y_test, y_pred):>13.1%} "
          f"{(y_test != y_pred).sum():>14} {cv_acc:>12.1%}")
    return y_pred


print("\n[2c  MODEL COMPARISON]")
print(f"{'Model':<22} {'Test accuracy':>13} {'Wrong (of 45)':>14} {'CV accuracy':>12}")
evaluate("Logistic regression", LogisticRegression(max_iter=1000))

# One confusion matrix per k-NN setting: rows = weights, columns = k
fig, axes = plt.subplots(len(WEIGHTS), len(SHOWN_K), figsize=(19, 7.5))
for row, w in zip(axes, WEIGHTS):
    for ax, k in zip(row, SHOWN_K):
        y_pred = evaluate(f"k-NN {w}, k={k}", KNeighborsClassifier(n_neighbors=k, weights=w))
        ConfusionMatrixDisplay.from_predictions(
            y_test, y_pred, display_labels=iris.target_names, cmap="Blues", colorbar=False, ax=ax)
        ax.set_title(f"{w} weights, k = {k}")
        ax.tick_params(axis="x", rotation=45)
fig.suptitle("Figure 7: Confusion matrices for k-NN (test set)")
fig.tight_layout(); fig.savefig("figure7_knn_confusion_matrices.png", dpi=200); plt.close(fig)
