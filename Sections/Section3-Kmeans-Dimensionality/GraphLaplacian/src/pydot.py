# -*- coding: utf-8 -*-
"""
pydot example 1
@author: Federico Cáceres
@url: http://pythonhaven.wordpress.com/2009/12/09/generating_graphs_with_pydot
"""
import pydot
graph = pydot.Dot(graph_type='graph')
added = set([])

def addNode(idx):
    if idx in added:
        return
    added.add(idx)
    (pred, fa, depth, (dim, res)) = trees[idx]
    if dim < 0:
        node1 = pydot.Node(shape="box")
        node1.set_name(str(idx))
        node1.set_label("root")

        label2 = "%.2f" % pred
        if pred > 0:
            label2 = '+' + label2
        node2 = pydot.Node()
        node2.set_name("s" + str(idx))
        node2.set_label(label2)

        graph.add_node(node1)
        graph.add_node(node2)
        edge = pydot.Edge(node1, node2, dir="forward")
        edge.set_label("Y")
        graph.add_edge(edge)
        return

    (pos, base) = posMap[dim]
    if pos > 0:
        posval = '+%.2f' % pos
    else:
        posval = pos
    label1 = "Position " + str(posval) + " is " + base
    label2 = "%.2f" % pred
    if pred > 0:
        label2 = '+' + label2

    node1 = pydot.Node(shape="box")
    node1.set_name(str(idx))
    node1.set_label(label1)

    node2 = pydot.Node()
    node2.set_name("s" + str(idx))
    node2.set_label(label2)

    graph.add_node(node1)
    graph.add_node(node2)
    edge = pydot.Edge(node1, node2, dir="forward")
    if t[-1][1]:
        edge.set_label("Y")
    else:
        edge.set_label("N")
    graph.add_edge(edge)
