from sklearn.datasets import load_breast_cancer
import numpy as np
import matplotlib.pyplot as plt

breast_cancer = load_breast_cancer()

X = breast_cancer.data
Y = breast_cancer.target

feature_names = breast_cancer.feature_names
target_name = breast_cancer.target_names

print("feature names:", feature_names)
print("target name:", target_name)

print("X shape:", X.shape)
print("Y shape:", Y.shape)

print("X sample names and Y sample:", feature_names, target_name[Y[0]])

print("Y distribution:", sum(Y == 0), "malignant,", sum(Y == 1), "benign")
print("Y distribution:", sum(Y == 0)/len(Y) * 100, "malignant,", sum(Y == 1)/len(Y) * 100, "benign")


print("Missing values in X and Y:", np.isnan(X).sum(), np.isnan(Y).sum())
print("Min and Max values in X mean radius:", np.min(X[:,0]), np.max(X[:,0]))

plt.hist(X[Y==0,0], bins=20, alpha=0.5, label='malignant')
plt.hist(X[Y==1,0], bins=20, alpha=0.5, label='benign')
plt.xlabel('Mean Radius')
plt.ylabel('Frequency')
plt.title('Distribution of Mean Radius by Diagnosis')
plt.legend()
plt.show()
