# Import necessary libraries
import numpy as np
from sklearn.metrics import fbeta_score


# Custom scorer function to evaluate performance of a model given the asymmetric cost
def custom_scorer(y_test, y_pred, cost_fp=1, cost_fn=1):
    # y_true: The true labels    
    # y_pred: The predicted labels
    # cost_fp: The cost of a false positive (wasting a call) 
    # cost_fn: The cost of a false negative (missing an opportunity)

    # Calculate the number of true positives, true negatives, false positives, and false negatives
    beta_asymmetric = np.sqrt(cost_fn / cost_fp)

    # Evaluate the model using the custom scorer
    score = fbeta_score(y_test, y_pred, beta=beta_asymmetric)
    return score
