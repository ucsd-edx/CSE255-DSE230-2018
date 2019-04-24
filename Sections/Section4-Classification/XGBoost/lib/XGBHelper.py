import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt

def visualize_features(bst, features_map = None):
    xgb.plot_importance(bst)
    plt.show()
    if features_map:
        print("Feature Mapping:")
        for x in features_map:
            print(x, "\t: ", features_map[x])

def get_error_values(y_pred, y_test, thresholds):
    accuracy_1 = []
    accuracy_0 = []
    for thresh in thresholds:
        y_test_i = y_test[y_test == 1]
        y_pred_i = y_pred[y_test == 1]
        correct = np.sum(y_pred_i > thresh)
        accuracy_1.append(1.0 * correct / len(y_test_i))

        y_test_i = y_test[y_test == 0]
        y_pred_i = y_pred[y_test == 0]
        correct = np.sum(y_pred_i <= thresh)
        accuracy_0.append(1.0 * correct / len(y_test_i))
    
    error_1 = list(1 - np.array(accuracy_1))
    error_0 = list(1 - np.array(accuracy_0))
    return error_1, error_0

def get_margin_plot(error_1, error_0, thresholds, legends = None, title=None, style=['b', 'r']):
    plt.plot(thresholds/(np.max(thresholds) - np.min(thresholds)), error_1, style[0])
    plt.plot(thresholds/(np.max(thresholds) - np.min(thresholds)), error_0, style[1])
    if legends:
        plt.legend(legends)
    plt.xlabel('Margin Score')
    plt.ylabel('Error %')
    if title:
        plt.title(title)

def statistics(y_pred, y_test, thr_lower, thr_upper):
    true_index = y_pred > thr_upper
    y_true =  np.sum(y_test[true_index] == 1)
    
    false_index = y_pred < thr_lower
    y_false =  np.sum(y_test[false_index] == 0)
    
    abstain = 1 - np.sum((y_pred < thr_lower) | (y_pred > thr_upper))/len(y_test)
    
    return (y_true+ y_false)/(len(y_test[true_index])+len(y_test[false_index])) , y_true/len(y_test[true_index]), y_false/len(y_test[false_index]), abstain