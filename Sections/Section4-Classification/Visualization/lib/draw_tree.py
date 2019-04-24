import pydotplus
from sklearn.externals.six import StringIO  
from sklearn.tree import export_graphviz
import re

def remove_gini_impurity(graph_val):
    """ Helper function """
    return re.sub(r'gini = (.*)<br/>s', "s", graph_val)

def draw_tree(dtree):
    """
    Takes an sklearn decision tree as input and generates a pydotplus graph
    returns a png object that can be used in IPython.display.Image 
    """
    dot_data = StringIO()
    export_graphviz(dtree, out_file=dot_data, filled=True, rounded=True, special_characters=True)
    graph_val = remove_gini_impurity(dot_data.getvalue())
    graph = pydotplus.graph_from_dot_data(graph_val) 
    return graph.create_png()