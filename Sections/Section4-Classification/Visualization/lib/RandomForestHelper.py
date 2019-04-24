import re
import sklearn.datasets as datasets
import pandas as pd
import pydotplus
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.externals.six import StringIO  
from IPython.display import Image  
from sklearn.tree import export_graphviz

def predict_category(rf, sample):
    '''
    Generates the predictions of each decision tree in the random forest, along with the overall prediction
    '''
    pred_list = []
    for child in rf.estimators_:
        pred_list.append(child.predict(sample))
    dtree_pred = np.array(pred_list, dtype=int).T
    final_pred = np.array(rf.predict(sample), dtype=int).reshape(len(sample), 1)
    return dtree_pred, final_pred

def compute_points(p1, p2, p3, A, B):
    '''
    Returns a co-ordinates of a point to plot on a simplex, where the point is defined by the equation p1*X1 + p2*X2 + p3*X3
    Note: It plots inside the triangle formed by the points (x1, y1), (x2, y2) and (x3, y3)
    '''
    return 1.0 * (p1*A[0] + p2*A[1] + p3*A[2]), 1.0 * (p1*B[0] + p2*B[1] + p3*B[2])

def get_rf_predictions(p1, p2, p3, label, A, B):
    '''
    Given the probabilities of each category, it generates a set of points to plot according to category
    '''
    x_list_0 = []
    y_list_0 = []
    x_list_1 = []
    y_list_1 = []
    x_list_2 = []
    y_list_2 = []
    pred_list = []
    for i in range(len(label)):
        x, y = compute_points(p1[i], p2[i], p3[i], A, B)
        #Classifying it into the right category for plotting points
        if p1[i] > p2[i] and p1[i] > p3[i]:
            x_list_0.append(x)
            y_list_0.append(y)
        elif p2[i] > p1[i] and p2[i] > p3[i]:
            x_list_1.append(x)
            y_list_1.append(y)
        elif p3[i] > p2[i] and p3[i] > p1[i]:
            x_list_2.append(x)
            y_list_2.append(y)
    return x_list_0, y_list_0, x_list_1, y_list_1, x_list_2, y_list_2

