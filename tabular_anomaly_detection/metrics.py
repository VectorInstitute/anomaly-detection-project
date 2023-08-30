from sklearn.metrics import roc_curve, roc_auc_score
from aequitas.group import Group
import pandas as pd

def calculate_tpr_at_fpr(y_test, scored_test, fpr_lim=0.05):
    """
    Calculate the model metrics, including TPR, FPR, threshold and AUROC based on a given FPR limit.

    Args:
    - scored_test (np.array): The predicted scores on the test set.
    - y_test (pd.DataFrame): The test set target variable.
    - fpr_lim (float): The FPR limit to use for obtaining the threshold and TPR. Default is 0.05.

    Returns:
    - metrics_dict (dict): A dictionary containing the TPR, FPR, threshold and AUROC.
    """
    # Calculate the ROC curve points
    fpr, tpr, threshold = roc_curve(y_test, scored_test)

    # Obtain the threshold and TPR based on the FPR
    obtained_tpr = tpr[fpr < fpr_lim][-1]
    obtained_threshold = threshold[fpr < fpr_lim][-1]
    obtained_fpr = fpr[fpr < fpr_lim][-1]

    # Calculate AUROC
    auroc = roc_auc_score(y_test, scored_test)

    # Store the metrics in a dictionary
    metrics_dict = {
        'TPR': round(obtained_tpr, 4),
        'FPR': round(obtained_fpr, 4),
        'Threshold': round(obtained_threshold, 4),
        'AUROC': round(auroc, 4),
    }

    return metrics_dict


def calculate_fairness_metrics(y_test, scored_test, X_test, fpr_lim=0.05):
    """
    Calculate fairness metrics on the predictions.

    Args:
    - scored_test (np.array): The predicted scores on the test set.
    - y_test (pd.Series): The test set target variable.
    - X_test (pd.DataFrame): The test set features.

    Returns:
    - fairness_ratio (float): The fairness ratio.
    """
    # Initialize the fairness evaluator
    g = Group()

    # Calculate the ROC curve points
    fpr, tpr, threshold = roc_curve(y_test, scored_test)

    # Obtain the threshold
    obtained_threshold = threshold[fpr < fpr_lim][-1]

    # Create a DataFrame with the scores, labels, and age
    df = pd.DataFrame({
        "score": scored_test,
        "label_value": y_test,
        "age": (X_test["customer_age"] > 50).map({True: ">50", False: "<=50"})
    })

    # Calculate the fairness metrics
    fairness_metrics = g.get_crosstabs(df, score_thresholds={"score_val": [obtained_threshold]})[0]

    # Calculate the fairness ratio as the min over max in FPR
    fairness_ratio = fairness_metrics["fpr"].min() / fairness_metrics["fpr"].max()

    return round(fairness_ratio, 4)