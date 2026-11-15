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
print(f"Test accuracy: {accuracy_score(y_test, y_pred):.3f} "
      f"({(y_test == y_pred).sum()} of {len(y_test)} flowers correct)")

ConfusionMatrixDisplay.from_predictions(
    y_test, y_pred, display_labels=iris.target_names, cmap="Blues", colorbar=False)
plt.title("Figure 5: Confusion matrix, logistic regression (test set)")
plt.tight_layout()
plt.savefig("figure5_confusion_matrix_logistic_regression.png", dpi=200)
plt.close()
