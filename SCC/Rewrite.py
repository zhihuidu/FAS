import pandas as pd
import networkx as nx
from networkx.algorithms.cycles import simple_cycles
import gurobipy as gp
from gurobipy import GRB
import sys
from datetime import datetime
import time
import csv
import random
from itertools import permutations
import os


#given a graph file in the csv format (each line is (source,destination, weight)), generate the graph data structure

def cleanfile(csv_file_path):
    node_list = set()
    edges = []
    with open(csv_file_path, mode='r') as csvfile:
        csvreader = csv.reader(csvfile)
        next(csvreader)

        for row in csvreader:
            source, target, weight = row
            weight = int(weight) #only weight should be integer, vertex ID can be a string
            if source != target :
                edges.append((source, target, weight))
            node_list.add(source)
            node_list.add(target)

    node_list = list(node_list)

    #here we merge multiple edges
    merged_edges = {}
    for source, dest, weight in edges:
        if (source, dest) in merged_edges:
            merged_edges[(source, dest)] += weight
        else:
            merged_edges[(source, dest)] = weight

    outputfile="Clean-"+os.path.basename(csv_file_path)
    with open(outputfile, 'w') as f:
        f.write(f"source,dest,weight\n")
        for source, dest in merged_edges:
            f.write(f"{source},{dest},{merged_edges[(source,dest)]}\n")

file_path = sys.argv[1]
cleanfile(file_path)

