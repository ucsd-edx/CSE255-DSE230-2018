import re
import sklearn.datasets as datasets
import pandas as pd
import pydotplus
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.externals.six import StringIO  
from IPython.display import Image  
from sklearn.tree import export_graphviz
from lib.RandomForestHelper import *
import matplotlib.pyplot as plt

def clean_dotty_data(graph_val):
    '''
    Removes the gini impurity field from the tree to generate a clean graph
    '''
    return re.sub(r'gini = (.*)<br/>s', "s", graph_val)

def generate_tree(rf, tree_index=0, height=0, width=0):
    '''
    Generates a pictorial representation of a decision tree
    '''
    dot_data = StringIO()
    export_graphviz(rf.estimators_[tree_index], out_file=dot_data, filled=True, rounded=True, special_characters=True)
    graph_val = clean_dotty_data(dot_data.getvalue())
    graph = pydotplus.graph_from_dot_data(graph_val) 
    print("Decision Tree", tree_index, ": ")
    if height > 0 and width > 0:
        display(Image(graph.create_png(), height=height, width=width))
    else:
        display(Image(graph.create_png()))

def generate_html(rf, sample, pred):
    '''
    Generates a html table with the predictions of all decision trees for each sample for the given dataset
    '''
    n_estimators = len(rf)
    sample_list = list(sample.index)
    dtree_list = []
    for i in range(n_estimators):
        dtree_list.append("DTree " + str(i))
    dtree_list.append("Prediction")
    pred_df = pd.DataFrame(pred, index=sample_list, columns=dtree_list)
    html = pred_df.to_html()
    return html

def generate_simplex(p, true, mistakes, labels,_figsize=(10,8)):
    """
    generate scatter of votes from Random Forests inside a simplex
    
    Params:
           p = a matrix with shape [no of examples X 3] representing the location of the points on the simplex.
           true: the true label for each example (0,1,2)
           mistakes: examples on which the majority prediction is incorrect.
           labels: the class name corresponding to each of the three labels.
    """
    A = [0, 1, 0.5] # the x and y coordinates of the corners of the simplex
    B = [0, 0, 0.87]

    fig = plt.figure(figsize=_figsize)
    ax = fig.add_subplot(111)

    plt.plot(A+A[0:1],B+B[0:1]) # draw the triangle
    size=np.ones(true.shape)
    size[mistakes]=20

    A=np.array(A); B=np.array(B)  # 
    # compute the locations of the points
    X=np.dot(p,A)
    Y=np.dot(p,B)
    rgb=['r','g','b']
    colors=[rgb[x] for x in true]
    plt.scatter(X,Y,c=colors,s=size)

    #Draw boundaries
    corner=np.array([1,0,0])
    margin=np.array([[.5,.5,0],[1./3,1./3,1./3],[.5,0,.5]])
    for alpha in np.arange(0.0,1.1,0.2):
        for rotate in range(3):
            boundry=alpha*margin+(1-alpha)*corner
            boundry=np.roll(boundry,rotate,axis=1)
            color='g:'
            if alpha==1:
                color='k:'
            plt.plot(np.dot(boundry,A),np.dot(boundry,B),color);
        
    plt.text(A[0]-0.05,B[0],str(labels[0]), fontsize=20, bbox=dict(facecolor='r', alpha=0.2))
    plt.text(A[1]+0.02,B[1],str(labels[1]), fontsize=20, bbox=dict(facecolor='g', alpha=0.2))
    plt.text(A[2]-0.015,B[2]+0.04,str(labels[2]), fontsize=20, bbox=dict(facecolor='b', alpha=0.2))
    plt.axis('off')
    plt.show()
