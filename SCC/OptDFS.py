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


FileNameHead="Clean-SCC-DFS"

#given a graph file in the csv format (each line is (source,destination, weight)), generate the graph data structure

def create_adjacency_lists(csv_file_path):
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

def build_dag(edge_weights,edge_flag):
    G = nx.DiGraph()
    for (u,v) in edge_weights :
         if edge_flag[(u,v)]==1:
              G.add_edge(u,v,weight=edge_weights[(u,v)])
    return G

def build_graph(edge_weights):
    G = nx.DiGraph()
    for (u,v) in edge_weights :
        G.add_edge(u,v,weight=edge_weights[(u,v)])
    return G

# Relabel the vertices in the DAG
def relabel_dag(G_dag):
    topological_order = list(nx.topological_sort(G_dag))
    mapping = {node: i for i, node in enumerate(topological_order)}
    return mapping

# Write the original vertex ID and its relative order to a file
def write_relabelled_nodes_to_file(mapping, output_file):
    with open(output_file, 'w') as f:
        for node, order in mapping.items():
            f.write(f"{node},{order}\n")

num=0




def calculate_updated_weight(nodes,edge_weights,out_adj,edge_flag,updated_weights):

    def dfs_iterative(start_node):
        stack = [(start_node, start_node, [])]  # (first, current_node, path_stack)

        while stack:
            first, node, path_stack = stack.pop()
            path_stack.append(node)

            for neighbor, weight in out_adj[node]:
                if neighbor < first:
                    continue

                if neighbor == first:
                    if edge_flag[(node, neighbor)] == 0:
                        continue

                    cycle = path_stack[:] + [neighbor]
                    cycle_edges = []
                    skip = False

                    for i in range(len(cycle) - 1):
                        if edge_flag[(cycle[i], cycle[i + 1])] == 1:
                            cycle_edges.append((cycle[i], cycle[i + 1]))
                        else:
                            skip = True
                            break

                    if not skip:
                        cycle_weights = []
                        for u, v in cycle_edges:
                            cycle_weights.append((u, v, edge_weights[(u, v)]))
                        min_edge = min(cycle_weights, key=lambda x: x[2])
                        for u, v, w in cycle_weights:
                            updated_weights[(u, v)] -= min_edge[2]

                else:
                    stack.append((first, neighbor, path_stack[:]))

            path_stack.pop()

    for node in  nodes: 
        print(f"update weight start from node {node} and we have {len(nodes)} nodes now")
        print(f"enter dfs")
        dfs_iterative(node)


def sccdfs_remove_cycle_edges(nodes,edge_weights,out_adj,edge_flag):
    oldnum=num
    removed_weight=0
    def dfs_iterative(start_node):
        global num
        nonlocal removed_weight
        stack = [(start_node, start_node, [])]  # (first, current_node, path_stack)

        while stack:
            first, node, path_stack = stack.pop()
            path_stack.append(node)

            for neighbor, weight in out_adj[node]:
                if neighbor < first:
                    continue

                if neighbor == first:
                    if edge_flag[(node, neighbor)] == 0:
                        continue

                    cycle = path_stack[:] + [neighbor]
                    cycle_edges = []
                    skip = False
                    for i in range(len(cycle) - 1):
                        if edge_flag[(cycle[i], cycle[i + 1])] == 1:
                            cycle_edges.append((cycle[i], cycle[i + 1]))
                        else:
                            skip = True
                            break
                    if not skip:
                        cycle_weights = []
                        for u, v in cycle_edges:
                            cycle_weights.append((u, v, updated_weights[(u, v)]))
                        min_edge = min(cycle_weights, key=lambda x: x[2])
                        removed_weight+=edge_weights[(min_edge[0],min_edge[1])]
                        print(f"removed {num+1} edges= {min_edge}")
                        num=num+1
                        edge_flag[(min_edge[0],min_edge[1])]=0

                else:
                    stack.append((first, neighbor, path_stack[:]))

            path_stack.pop()


    updated_weights=edge_weights.copy()
    print(f"first update the weight in different cycles")
    calculate_updated_weight(nodes,edge_weights,out_adj,edge_flag,updated_weights)
    print(f"end of update the weight in different cycles")

    removed_edges = set()
    rec_stack = set()
    removed_weight=0
    for node in nodes: 
        print(f"remove start from node {node} and we have {len(nodes)} nodes now")
        print(f"enter dfs")
        dfs_iterative(node)


    return removed_weight 

def process_graph(file_path):
    global num
    print(f"read data")
    node_list, edge_weights, in_edges, out_edges = create_adjacency_lists(file_path)
    G=build_graph(edge_weights)
    total=sum(edge_weights[(u,v)] for (u,v) in edge_weights)
    print(f"total number of nodes={len(node_list)}, total number of edges={len(edge_weights)}")
    print(f"sum of weight={total}")

    removed_weight=0
    edge_flag={(u,v):1 for (u,v) in edge_weights }
    shG=G.copy()

    while not nx.is_directed_acyclic_graph(shG):
        print("the graph is not a DAG.")
        component=node_list.copy()
        removed_weight += sccdfs_remove_cycle_edges(component, edge_weights,out_edges,edge_flag )
        print(f"totally removed {removed_weight}, percentage is {removed_weight/total*100}\n\n")
        removednum=0
        for u,v in edge_flag:
            if edge_flag[(u,v)]==0:
                    removednum+=1
        print(f"to here removed {removednum} edges")

        print(f"sum of removed weight={removed_weight},percentage of remained  weight ={(total-removed_weight)/total *100}")

        print(f"build dag")
        shG=build_dag(edge_weights,edge_flag)
        dag_weight=sum(data['weight'] for u, v, data in shG.edges(data=True))
        print(f"dag weight={dag_weight}")
        if dag_weight+removed_weight != total:
            print("something wrong with the removed edge weights, remained weights, and the total weights")



    print(f"relabel dag")
    mapping = relabel_dag(shG)

    print(f"write file")
    current_time = datetime.now()
    time_string = current_time.strftime("%Y%m%d_%H%M%S")
    output_file = f"{FileNameHead}-Relabel-{time_string}.csv"
    write_relabelled_nodes_to_file(mapping, output_file)




file_path = sys.argv[1]
sys.setrecursionlimit(990000000)
process_graph(file_path)

