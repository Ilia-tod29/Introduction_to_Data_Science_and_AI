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
from sklearn.model_selection import train_test_split
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
SHOWN_K = [1, 5, 15, 50, 75, 105]         # values of k shown in the table

# accuracy[weights][k - 1] = (training accuracy, test accuracy)
accuracy = {w: [] for w in WEIGHTS}
for w in WEIGHTS:
    for k in K_VALUES:
        knn = KNeighborsClassifier(n_neighbors=k, weights=w).fit(X_train, y_train)
        accuracy[w].append((knn.score(X_train, y_train), knn.score(X_test, y_test)))

print("\n[2b  K-NEAREST NEIGHBOURS]  accuracy (train / test)")
print(f"{'k':>4}   {'uniform':^15} {'distance':^15}")
for k in SHOWN_K:
    cells = [f"{train:.3f} / {test:.3f}" for train, test in (accuracy[w][k - 1] for w in WEIGHTS)]
    print(f"{k:>4}   {cells[0]:^15} {cells[1]:^15}")

# Test accuracy against k, with logistic regression as a reference line
fig, ax = plt.subplots(figsize=(8, 5.5))
for w in WEIGHTS:
    ax.plot(K_VALUES, [test for _, test in accuracy[w]], label=f"k-NN, {w} weights")
ax.axhline(log_reg_acc, color="gray", ls="--", label="Logistic regression")
ax.set(xlabel="Number of neighbours (k)", ylabel="Test accuracy",
       title="Figure 6: k-NN test accuracy for growing k")
ax.grid(alpha=0.3); ax.legend()
fig.tight_layout(); fig.savefig("figure6_knn_accuracy_vs_k.png", dpi=200); plt.close(fig)
