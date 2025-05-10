import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

def plot_confusion_matrix(y_true, y_pred, labels):
    """
    Plot a confusion matrix using seaborn heatmap.
    Args:
        y_true (np.array): True labels (class indices).
        y_pred (np.array): Predicted labels (class indices).
        labels (list): Class names.
    Returns:
        None: Displays the plot.
    Note:
        Used for ESC-10 classification to visualize misclassifications.
    """
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots()
    sns.heatmap(cm, square=True, annot=True, annot_kws={'fontsize': 16}, fmt="d", cmap='Blues', cbar=True, ax=ax)
    ax.set_xlabel('Predicted labels')
    ax.set_ylabel('True labels')
    ax.set_title('Confusion Matrix')
    ax.xaxis.set_ticklabels(labels)
    ax.yaxis.set_ticklabels(labels)
    return fig

def generate_classification_report(y_true, y_pred, class_names):
    """
    Generate classification report for model predictions.
    Args:
        y_true (np.array): True labels (one-hot or indices).
        y_pred (np.array): Predicted labels (probabilities or indices).
        class_names (list): Class names.
    Returns:
        report (str): Classification report.
    """
    y_true = np.argmax(y_true, axis=1) if y_true.ndim > 1 else y_true
    y_pred = np.argmax(y_pred, axis=1) if y_pred.ndim > 1 else y_pred
    return classification_report(y_true, y_pred, target_names=class_names)