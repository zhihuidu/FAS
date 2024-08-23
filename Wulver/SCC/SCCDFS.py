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


FileNameHead="SCC-DFS"

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

def build_new_shrunk_graph (G,edge_flag):
    shrunkG = nx.DiGraph()
    for u, v, data in G.edges(data=True):
         if edge_flag[(u,v)]==1:
              shrunkG.add_edge(u,v,weight=data['weight'])
    return shrunkG

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

def read_ID_Mapping(file_path):
    df=pd.read_csv(file_path, header=None, names=['ID', 'Index'])
    data_table = df.values.tolist()
    data_dict = {key: value for key, value in data_table}
    return data_dict

num=0
def sccdfs_remove_cycle_edges(nodes,edge_weights,out_adj,edge_flag):
    oldnum=num
    removed_weight=0
    def dfs(node, stack, rec_stack ):
        global num
        nonlocal removed_weight
        rec_stack.add(node)
        stack.append(node)
        for neighbor, weight in out_adj[node]:
            if neighbor not in nodes :
                continue

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
                    removed_edges.add(min_edge)
                    print(f"removed {num+1} edges= {min_edge}")
                    num=num+1
                    edge_flag[(min_edge[0],min_edge[1])]=0
            else:
                    dfs(neighbor, stack, rec_stack)
                    if neighbor in nodes:
                         nodes.remove(neighbor)
        restorenode=stack.pop()
        rec_stack.remove(node)
        if node in nodes:
            nodes.remove(node)
        return False
    
    removed_edges = set()
    rec_stack = set()
    removed_weight=0
    if len(nodes)>0: 
        node = nodes[0]
        print(f"start from node {node} and we have {len(nodes)} nodes now")
        print(f"enter dfs")
        dfs(node, [], rec_stack)


    return removed_weight, removed_edges

def again_dfs_remove_cycle_edges(nodes,edge_weights,out_adj,edge_flag):
    removed_edges = set()
    def dfs(node, stack, visited, rec_stack):
        global num
        nonlocal removed_weight
        nonlocal tovisit
        visited.add(node)
        rec_stack.add(node)
        stack.append(node)
        for neighbor, weight in out_adj[node]:
            if neighbor not in visited:
                if neighbor in tovisit:
                    tovisit.remove(neighbor)
                dfs(neighbor, stack, visited, rec_stack)
            elif neighbor in rec_stack and edge_flag[(node, neighbor)]==1:
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
                    #removed_edges.add(min_edge)
                    print(f"removed {num+1} edges {min_edge}")
                    num=num+1
                    edge_flag[(min_edge[0],min_edge[1])]=0
        rec_stack.remove(node)
        restorenode=stack.pop()
        #visited.remove(restorenode)
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

    print(f"after read removed edges, weight={removed_weight}, remained={41912141-removed_weight} dfs again")
    iternum=0
    new_removed_weight=removed_weight
    while iternum ==0 :
        visited = set()
        rec_stack = set()
        tovisit =nodes.copy()
        random.shuffle(tovisit)
        while len(tovisit)>0:
            node = tovisit.pop(0)
            print(f"start from node {node} and we have {len(tovisit)} nodes now")
            if node not in visited:
                dfs(node, [], visited, rec_stack)
        new_removed_weight = sum(w for u, v,w in removed_edges)
        removed_weight += new_removed_weight
        iternum=iternum+1
    print("finish  again bfs")
    return removed_weight, removed_edges



def process_graph(file_path):
    print(f"read data")
    node_list, edge_weights, in_edges, out_edges = create_adjacency_lists(file_path)
    G=build_graph(edge_weights)
    total=sum(edge_weights[(u,v)] for (u,v) in edge_weights)
    print(f"total number of nodes={len(node_list)}, total number of edges={len(edge_weights)}")
    print(f"sum of weight={total}")

    removed_weight=0
    edge_flag={(u,v):1 for (u,v) in edge_weights }
    removed_weight,removed_edges=again_dfs_remove_cycle_edges(node_list, edge_weights,out_edges,edge_flag )

    shG=build_new_shrunk_graph (G,edge_flag)



    while not nx.is_directed_acyclic_graph(shG):
        print("the graph is not a DAG.")
        print(f"strongly connected components")
        scc=list(nx.strongly_connected_components(shG))
        #print(f"scc={scc}")
        numcomponent=0
        for component in scc:
            if len(component)==1:
                 continue
            print(f"handle the {numcomponent}th component")
            removed_weight1,removed_edges=sccdfs_remove_cycle_edges(list(component), edge_weights,out_edges,edge_flag )
            removed_weight+=removed_weight1
            print(f"removed {len(removed_edges)} edges and their weight is {removed_weight1}, totally removed {removed_weight}, percentage is {removed_weight/total*100}\n\n")
            numcomponent+=1


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
sys.setrecursionlimit(900000)
process_graph(file_path)

