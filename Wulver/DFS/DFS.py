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


FileNameHead="DFS"

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



def read_ID_Mapping(file_path):
    df=pd.read_csv(file_path, header=None, names=['ID', 'Index'])
    data_table = df.values.tolist()
    data_dict = {key: value for key, value in data_table}
    return data_dict

num=0
def dfs_remove_cycle_edges(nodes,edge_weights,out_adj,edge_flag):
    oldnum=num
    def dfs(node, stack, rec_stack ):
        global num
        nonlocal removed_weight
        stack.append(node)
        rec_stack.add(node)
        for neighbor, weight in out_adj[node]:
            if neighbor in rec_stack :
                if edge_flag[(node, neighbor)]==0:
                      continue
                #print(f"handle visited neighbor {neighbor}")
                cycle = stack[stack.index(neighbor):] + [neighbor]
                cycle_edges=[]
                skip=0
                for i in range(len(cycle) - 1) :
                     if edge_flag[(cycle[i], cycle[i + 1])]==1:
                          cycle_edges.append((cycle[i], cycle[i + 1]))      
                     else:
                         skip=1
                         break
                if skip==0:
                    #print(f"find cycle {cycle_edges}")
                    cycle_weights=[]
                    for u, v in cycle_edges:
                        cycle_weights.append((u,v,edge_weights[(u,v)]))
                    min_edge = min(cycle_weights, key=lambda x: x[2])
                    removed_weight+=min_edge[2]
                    #removed_edges.add(min_edge)
                    print(f"removed {num+1} edges= {min_edge}")
                    num=num+1
                    edge_flag[(min_edge[0],min_edge[1])]=0
            else :
                dfs(neighbor, stack, rec_stack)
                if neighbor in tovisit:
                      tovisit.remove(neighbor)
        restorenode=stack.pop()
        rec_stack.remove(node)
        if node in tovisit:
             tovisit.remove(node)
        return False
    
    #removed_weight = sum(w for u, v,w in removed_edges)
    removed_edges = set()
    rec_stack = set()
    removed_weight=0
    tovisit=nodes.copy()
    while len(tovisit)>0: 
        node = tovisit.pop(0)
        print(f"start from node {node} and we have {len(tovisit)} nodes now")
        dfs(node, [], rec_stack)


    return removed_weight, removed_edges

def again_dfs_remove_cycle_edges(nodes,edge_weights,out_adj,edge_flag):
    removed_edges = set()
    def dfs(node, stack, rec_stack):
        global num
        nonlocal removed_weight
        rec_stack.add(node)
        stack.append(node)
        for neighbor, weight in out_adj[node]:
            if neighbor in rec_stack :
                if edge_flag[(node, neighbor)]==0:
                       continue
                cycle = stack[stack.index(neighbor):] + [neighbor]
                cycle_edges=[]
                skip=0
                for i in range(len(cycle) - 1) :
                     if edge_flag[(cycle[i], cycle[i + 1])]==1:
                          cycle_edges.append((cycle[i], cycle[i + 1]))      
                     else:
                         skip=1
                         break
                if skip==0:
                    cycle_weights=[]
                    for u, v in cycle_edges:
                        cycle_weights.append((u,v,edge_weights[(u,v)]))
                    min_edge = min(cycle_weights, key=lambda x: x[2])
                    removed_weight+=min_edge[2]
                    print(f"removed {num+1} edges {min_edge}")
                    num=num+1
                    edge_flag[(min_edge[0],min_edge[1])]=0
            else:
                dfs(neighbor, stack, rec_stack)
                if neighbor in tovisit:
                        tovisit.remove(neighbor)
        rec_stack.remove(node)
        restorenode=stack.pop()
        if node in tovisit:
              tovisit.remove(node)
        return False
    

    # Open the CSV file
    print(f"read removed edges file removed.txt")
    removed_weight=0
    with open('removed.txt', 'r') as file:
              reader = csv.reader(file)
              for row in reader:
                 x1,x2,x3  = row # Strip any leading/trailing whitespace
                 x11,x12=x1.split(maxsplit=1)
                 x12=int(x12)
                 x2=int(x2)
                 if edge_flag[(x12,x2)] ==1:
                     edge_flag[(x12,x2)] = 0
                     removed_weight+=edge_weights[(x12,x2)]

    print(f"after read removed edges, weight={removed_weight}, remained={41912141-removed_weight}")
    print(f"remained percentage={(41912141-removed_weight)/41912141.0*100}")
    print(f"dfs again")
    iternum=0
    while iternum ==0 :
        rec_stack = set()
        tovisit =nodes.copy()
        random.shuffle(tovisit)
        while len(tovisit)>0:
            node = tovisit[0]
            print(f"start from node {node} and we have {len(tovisit)} nodes now")
            dfs(node, [], rec_stack)

        iternum=iternum+1
    print("finish  again bfs")
    return removed_weight, removed_edges

def build_dag(edge_weights,edge_flag):
    G = nx.DiGraph()
    for (u,v) in edge_weights :
         if edge_flag[(u,v)]==1:
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

def process_graph(file_path):
    print(f"read data")
    node_list, edge_weights, in_edges, out_edges = create_adjacency_lists(file_path)
    total=sum(edge_weights[(u,v)] for (u,v) in edge_weights)
    print(f"total number of nodes={len(node_list)}, total number of edges={len(edge_weights)}")
    print(f"sum of weight={total}")

    print(f"dfs")
    edge_flag={(u,v):1 for (u,v) in edge_weights }
    
    print(f"enter again dfs")
    removed_weight,removed_edges=again_dfs_remove_cycle_edges(node_list, edge_weights,out_edges,edge_flag )
    #print(f"enter compete dfs ")
    #removed_weight1,removed_edges=dfs_remove_cycle_edges(node_list, edge_weights,out_edges,edge_flag )
    #removed_weight=removed_weight1
    print(f"sum of removed weight={removed_weight},percentage of remained  weight ={(total-removed_weight)/total *100}")

    print(f"build dag")
    DAG=build_dag(edge_weights,edge_flag)
    dag_weight=sum(data['weight'] for u, v, data in DAG.edges(data=True))
    print(f"dag weight={dag_weight}")
    if dag_weight+removed_weight != total:
        print("something wrong with the removed edge weights, remained weights, and the total weights")

    if not nx.is_directed_acyclic_graph(DAG):
        print("the graph is not a DAG. There is something wrong")

    print(f"relabel dag")
    mapping = relabel_dag(DAG)

    print(f"write file")
    current_time = datetime.now()
    time_string = current_time.strftime("%Y%m%d_%H%M%S")
    output_file = f"{FileNameHead}-Relabel-{time_string}.csv"
    write_relabelled_nodes_to_file(mapping, output_file)




file_path = sys.argv[1]
sys.setrecursionlimit(900000)
process_graph(file_path)

