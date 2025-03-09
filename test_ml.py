import pytest
# TODO: add necessary import
import numpy as np
import pandas as pd
from ml.data import process_data
from ml.model import compute_model_metrics, train_model
from sklearn.linear_model import LogisticRegression

## Dataset to pull from for unit tests
df = pd.DataFrame({
    "num_feature": [5, 15, 25, 35, 45],
    "cat_feature": [1, 0, 1, 0, 1],
    "target": [0, 1, 0, 1, 0]
})

## Process data 
X_train, y_train, _, _ = process_data(
    df,
    categorical_features=["cat_feature"],
    label="target",
    training=True,
    )

## Train model
model = train_model(X_train, y_train)


# TODO: implement the first test. Change the function name and input as needed
def test_model_default_solver():
    """
    Check that the returned model from train_model is lbfgs for LogisticRegression
    """
    
    # Your code here
    assert model.solver == 'lbfgs'
    pass


# TODO: implement the second test. Change the function name and input as needed
def test_model_instance_type():
    """
    # Checks to see that train_model returns LogisticRegression model instance
    """
    
    # Your code here
    assert isinstance(model, LogisticRegression)
    pass


# TODO: implement the third test. Change the function name and input as needed
def test_metrics_values():
    """
    Check that compute_model_metrics returns expected precision, recall,
    and F1 score for test arrays
    """
    
    # make two arrays 
    actual_labels = np.array([1, 0, 1, 0, 1])
    predicted_labels = np.array([1, 0, 0, 0, 1])
    
    # Calculate metrics compute_model_metrics function from ml/model 
    precision, recall, fbeta = compute_model_metrics(actual_labels, predicted_labels)
    
    # True positives = 2, false positives = 0, false negatives = 1
    # precision = 2/2 = 1.0, recall = 2/3 = 0.6667,
    # F-beta is computed as F1 score (beta=1) so fbeta = 0.8.
    expected_precision = 1.0
    expected_recall = 0.6667
    expected_fbeta = 0.8
    
    # Compare 
    assert round(precision, 4) == expected_precision
    assert round(recall, 4) == expected_recall
    assert round(fbeta, 4) == expected_fbeta
    pass
