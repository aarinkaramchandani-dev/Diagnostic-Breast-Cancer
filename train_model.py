from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
import numpy as np
import matplotlib.pyplot as plt

breast_cancer = load_breast_cancer()

X = breast_cancer.data
Y = breast_cancer.target

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42, stratify=Y)

print("Training dataset:", X_train.shape, Y_train.shape, "Testing dataset:", X_test.shape, Y_test.shape)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model = LogisticRegression(max_iter=10000)
model.fit(X_train, Y_train)

predictions = model.predict(X_test)
score = model.score(X_test, Y_test)

print("Model accuracy:", score)
print("First 10 values of X_test and Y_test:", X_test[:10], Y_test[:10])

confusion = confusion_matrix(Y_test, predictions, labels=[0, 1])
print("Confusion matrix:", confusion)

correct_malignant = confusion[0, 0] / (confusion[0,1] + confusion[0,0]) * 100
correct_benign = confusion[1, 1] / (confusion[1,0] + confusion[1,1]) * 100

display = ConfusionMatrixDisplay(
    confusion_matrix=confusion,
    display_labels=breast_cancer.target_names
)
display.plot()
plt.title(f"Confusion Matrix\nMalignant recall: {correct_malignant:.1f}%\nBenign recall: {correct_benign:.1f}%", fontsize=10)
plt.show()
