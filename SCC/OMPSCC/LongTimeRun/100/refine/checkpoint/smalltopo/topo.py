from collections import Counter
import pandas as pd
import networkx as nx
from networkx.algorithms.cycles import simple_cycles
import sys
import os
from datetime import datetime
import time
import csv
import random
from itertools import permutations
import numpy as np
from pathlib import Path

FileNameHead="topo"
Nt=10
Np=9000000

# Write the original vertex ID and its relative order to a file
def write_labelled_nodes_to_file(id_mapping, output_file):
    with open(output_file, 'w') as f:
        f.write(f"Node Id, Order\n")
        for node, order in id_mapping.items():  
            f.write(f"{node},{order}\n")

# Main function to perform all steps


def read_ID_Mapping(file_path):
    #df=pd.read_csv(file_path, header=None, names=['ID', 'Index'])
    df=pd.read_csv(file_path)
    data_table = df.values.tolist()
    data_dict = {key: value for key, value in data_table}
    return data_dict



#given a graph file in the csv format (each line is (source,destination, weight)), generate the graph data structure
def build_ArrayDataStructure(csv_file_path):
    node_list = set()
    edges = []
    with open(csv_file_path, mode='r') as csvfile:
        csvreader = csv.reader(csvfile)
        next(csvreader)

        for row in csvreader:
            source, target, weight = row
            source = int(source)
            target = int(target)
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

    in_adj={}
    out_adj={}
    for node in node_list:
        out_adj[node]=[]
        in_adj[node]=[]
    for source, target in merged_edges:
        out_adj[source].append((target, merged_edges[(source,target)]))
        in_adj[target].append((source, merged_edges[(source,target)]))

    return node_list, merged_edges, in_adj, out_adj 


def get_lvs_mvs_rvs(nodes, in_edges, out_edges):
    lvs = [node for node in nodes if len(in_edges[node]) == 0 and len(out_edges[node])>=0]
    #vertices in the left vertex list (lvs) should have small index without refining

    rvs = [node for node in nodes if len(out_edges[node]) == 0 and len(in_edges[node]) > 0]
    #vertices in the right vertex list (rvs) should have large index without refining

    mvs = [node for node in nodes if len(in_edges[node])>0 and len(out_edges[node])>0]
    #vertices in the middle vertex list (mvs) should have middle index and we need to refine them

    return lvs, mvs, rvs

def build_from_EdgeList(edge_weights):
    G = nx.DiGraph()
    for (u,v) in edge_weights :
        G.add_edge(u,v,weight=edge_weights[(u,v)])
    return G


def sum_of_weight_for_current_mapping (edges,id_mapping):
    w = 0
    for source, target in edges:
        if id_mapping [source] < id_mapping [target]:
            w += edges[(source,target)]
    return w


def build_feedforward_from_order(edge_weights,mapping):
    forwardedges=[]
    for (u,v) in edge_weights :
        if mapping[u]<mapping[v]:
            forwardedges.append((u,v,edge_weights[(u,v)]))
    return forwardedges


def random_topological_sort(graph):
    # Get the in-degree of each node
    in_degree = {node: 0 for node in graph.nodes()}
    for _, target in graph.edges():
        in_degree[target] += 1

    # Initialize a list to store the sorted order
    sorted_order = []
    # Create a list of nodes with zero in-degree
    zero_in_degree_nodes = [node for node in graph.nodes() if in_degree[node] == 0]

    while zero_in_degree_nodes:
        # Randomly shuffle the nodes with zero in-degree
        random.shuffle(zero_in_degree_nodes)
        # Pick the first node and remove it from the list
        current_node = zero_in_degree_nodes.pop(0)
        sorted_order.append(current_node)

        # Decrease the in-degree of the neighbors
        for neighbor in graph.neighbors(current_node):
            in_degree[neighbor] -= 1
            if in_degree[neighbor] == 0:
                zero_in_degree_nodes.append(neighbor)

    id_mapping = {node: idx for idx, node in enumerate(sorted_order)}
    return id_mapping

def calculate_score(G, order_mapping):
    """
    Calculate the sum of weights for edges where the source's order is less than the target's order.

    :param G: A directed graph with weighted edges.
    :param order_mapping: Dictionary mapping vertex ID to its order.
    :return: Sum of weights for the qualifying edges.
    """
    score = 0

    # Iterate through the edges of the graph
    for source, target, data in G.edges(data=True):
        weight = data.get('weight', 0)

        # Get the orders from the order mapping
        source_order = order_mapping[source]
        target_order = order_mapping[target]

        # Check if both orders exist and if the source's order is less than the target's order
        if source_order < target_order:
                score += weight

    return score

# Write the original vertex ID and its relative order to a file
def write_relabelled_nodes_to_file(mapping, output_file):
    with open(output_file, 'w') as f:
        f.write(f"{'Node ID'},{'Order'}\n")
        for node, order in mapping.items():
            f.write(f"{node},{order}\n")


def compute_change(nodes, in_adj, out_adj,id_mapping, new_id_mapping):
    #for the vertex ID list such as [1,50,10,3],we shuffle the IDs and calculate the best increased weights based on different mappings
    #new_id_mapping is used to calculate the gained weight for changed mapping

    delta = 0
    for node in nodes:
        for target, weight in out_adj[node]:
            if id_mapping[node] < id_mapping [target]:
                delta -= weight
            if new_id_mapping[node] < new_id_mapping[target]:
                delta += weight
        for source, weight in in_adj[node]:
            if source in nodes:
                continue
            if id_mapping [source] < id_mapping[node] :
                delta -= weight
            if new_id_mapping [source] < new_id_mapping[node] :
                delta += weight
    return delta

def listshuffle(vertexlist,in_edges,out_edges,id_mapping,new_id_mapping):
    # vertexlist is the ID list whose ID will be permuated to calculate the best weight sum 

    current_order = [id_mapping[idx] for idx in vertexlist]
    best_order = current_order.copy()
    best_delta = 0    
    perms = list(permutations(vertexlist))
    perms=perms[1:] #remove the original one without any changes
    for perm in perms:
        for index,node in enumerate (perm) :
              new_id_mapping[perm[index]]=id_mapping[vertexlist[index]]

        delta = compute_change(vertexlist, in_edges,out_edges,id_mapping, new_id_mapping)

        if delta > best_delta and delta > 0.00000001:
            best_delta = delta
            for index in range(len(perm)):
                best_order[index] = new_id_mapping[vertexlist[index]]
        
        for index,node in enumerate (perm) :
              new_id_mapping[vertexlist[index]]=id_mapping[vertexlist[index]]

    if best_delta >0 :
        for i, node in enumerate(vertexlist):
            id_mapping[node] = best_order[i]
            new_id_mapping[node]=best_order[i]

    return best_delta

def process_graph(file_path,theid_mapping):
    print(f"read data")
    node_list, edge_weights, in_edges, out_edges= build_ArrayDataStructure(file_path)
    lvs,mvs,rvs = get_lvs_mvs_rvs(node_list, in_edges,out_edges)
    G=build_from_EdgeList(edge_weights)
    total=sum(edge_weights[(u,v)] for (u,v) in edge_weights)
    print(f"total number of nodes={len(node_list)}, total number of edges={len(edge_weights)}")
    print(f"sum of weight={total}")
    id_mapping={}
    if len(theid_mapping) >0 :
        id_mapping=theid_mapping
    else:
        id_mapping = {node: idx for idx, node in enumerate(lvs + mvs + rvs)}
    current_score= sum_of_weight_for_current_mapping (edge_weights,id_mapping)
    print(f"current score is {current_score}, percentage is {current_score/total}")

    while True: 
        edges=build_feedforward_from_order(edge_weights,id_mapping)
        G_dag = nx.DiGraph()
        G_dag.add_nodes_from(node_list)
        G_dag.add_weighted_edges_from(edges)
        newmapping=random_topological_sort(G_dag)
        for i in range(Nt):
            print(f"the {i}(th) topological sorting")
            newscore=sum_of_weight_for_current_mapping (edge_weights,newmapping)
            if newscore>current_score:
                output_file = f"{newscore}.csv"
                write_relabelled_nodes_to_file(newmapping, output_file)
                current_score=newscore
                print(f"write new topological sorting score {current_score}, percentage is {current_score/total}")
                id_mapping=newmapping.copy()
            newmapping=random_topological_sort(G_dag)
                
        for i in range(Np):
                vertexlist = random.sample(mvs, 4)
                newmapping=id_mapping.copy()
                delta = listshuffle(vertexlist,in_edges,out_edges,id_mapping,newmapping)
                if delta >0:
                    current_score+=delta
                    output_file = f"{current_score}.csv"
                    write_relabelled_nodes_to_file(newmapping, output_file)
                    print(f"write new permutation score {newscore}")

        print(f"finished {Nt} topological sorting and {Np} permutationed")


file_path = sys.argv[1]
myid_mapping={}
print(f"{sys.argv[1]}")
print(f"{sys.argv[2]}")
if len(sys.argv) >2: 
    myid_mapping = read_ID_Mapping(sys.argv[2])
process_graph(file_path,myid_mapping)
