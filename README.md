# Diagnostic Breast Cancer Classifier

This is a learning project using the Breast Cancer Wisconsin (Diagnostic) dataset. I trained two logistic regression models to classify samples as **malignant** or **benign** from measurements of cell nuclei:

- `train_model.py` uses all 30 features.
- `train_model_paper.py` uses three features discussed in a [1994 paper by Mangasarian, Street, and Wolberg](https://cdn.aaai.org/Symposia/Spring/1994/SS-94-01/SS94-01-019.pdf).

The paper used a different classification method. My three-feature model is inspired by its feature choice, not a reproduction of its results.

## Data

The dataset contains **569 samples**: 212 malignant and 357 benign. Each sample has 30 numerical features derived from an image of a fine needle aspirate of a breast mass. These features describe properties of cell nuclei, such as radius, area, texture, and smoothness. The dataset is included with scikit-learn, so no separate download is needed.

The paper highlighted *extreme area*, *extreme smoothness*, and *mean texture*. Their names in scikit-learn are `worst area`, `worst smoothness`, and `mean texture` (columns 23, 24, and 1).

## What I did

1. Loaded the dataset and explored its shapes, class counts, feature values, missing values, and distributions.
2. Used the same stratified 80/20 train/test split for both models (`random_state=42`): 455 training samples and 114 test samples.
3. Fit `StandardScaler` on each model's training data, then used that scaler to transform its test data.
4. Trained `LogisticRegression` on the training data and checked its predictions, accuracy, and confusion matrix on the test data.

The models differ in **which features they receive**. The training and test split, scaling procedure, and model type are the same.

## Results from this one test split

| Metric | All 30 features | Three paper-inspired features |
| --- | ---: | ---: |
| Correct predictions | 112/114 | 108/114 |
| Accuracy | 98.2% | 94.7% |
| Malignant correctly identified | 41/42 | 40/42 |
| Benign correctly identified | 71/72 | 68/72 |
| Malignant cases predicted benign | 1 | 2 |
| Benign cases predicted malignant | 1 | 4 |

For the 30-feature model, the confusion matrix is `[[41, 1], [1, 71]]`. For the three-feature model, it is `[[40, 2], [4, 68]]`. Rows are actual diagnoses and columns are predicted diagnoses, ordered **malignant, benign**.

The 30-feature model performed better on this particular split. One split of 114 test samples is a limited comparison, so these numbers do not establish how either model would perform on new clinical data.

**This is a learning exercise, not a medical device. Do not use it to make diagnostic decisions.**

## Run it yourself

```bash
git clone https://github.com/aarinkaramchandani-dev/Diagnostic-Breast-Cancer.git
cd Diagnostic-Breast-Cancer
python3 -m venv .venv
source .venv/bin/activate
pip install scikit-learn matplotlib numpy

python inspect_data.py
python train_model.py
python train_model_paper.py
```

Each script prints its results and opens a plot window. Close the plot window to finish that script.

## Files

- `inspect_data.py`: explore the dataset.
- `train_model.py`: train and test the model using all 30 features.
- `train_model_paper.py`: train and test the model using the three selected features.

## Reference

Mangasarian, O. L., Street, W. N., and Wolberg, W. H. (1994). [*Breast Cancer Diagnosis and Prognosis via Linear Programming*](https://cdn.aaai.org/Symposia/Spring/1994/SS-94-01/SS94-01-019.pdf). AAAI Technical Report SS-94-01.
