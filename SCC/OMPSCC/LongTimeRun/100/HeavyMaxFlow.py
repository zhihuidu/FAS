import matplotlib.pyplot as plt
from collections import Counter
import pandas as pd
import networkx as nx
from networkx.algorithms.cycles import simple_cycles
import gurobipy as gp
from gurobipy import GRB
import sys
import os
from datetime import datetime
import time
import csv
import random
from itertools import permutations
import numpy as np
from sklearn.cluster import SpectralClustering
import matplotlib.pyplot as plt

FileNameHead="Heavy-SCC-OMP-DFS"



def read_num_pairs(file_path):
    numpairs=2
    mindistance=5
    if os.path.exists(file_path):
        with open(file_path,mode='r') as f:
            csv_reader = csv.reader(f)
            for row in csv_reader:
                numpairs=int(row[0])
                mindistance=int(row[1])
    return numpairs,mindistance


# Function to find the minimum cut using Edmonds-Karp algorithm
def find_minimum_cut(graph, source, target):
    # Compute the maximum flow and minimum cut
    cut_value, partition = nx.minimum_cut(graph, source, target, capacity='weight', flow_func=nx.algorithms.flow.edmonds_karp)

    reachable, non_reachable = partition
    cut_edges = []

    for u in reachable:
        for v in graph[u]:
            if v in non_reachable:
                cut_edges.append((u, v, graph.edges[u, v]['weight']))

    return cut_value, cut_edges




# Function to perform spectral clustering on the graph
def spectral_clustering_divide(graph, n_clusters=2):
    # Get the adjacency matrix with weights
    adj_matrix = nx.to_numpy_array(graph, weight='weight')

    # Perform spectral clustering
    sc = SpectralClustering(n_clusters=n_clusters, affinity='precomputed',
                            assign_labels='discretize', random_state=42)
    labels = sc.fit_predict(adj_matrix)

    # Create subgraphs for each cluster
    subgraphs = []
    for i in range(n_clusters):
        nodes_in_cluster = [node for node, label in enumerate(labels) if label == i]
        subgraph = graph.subgraph(nodes_in_cluster)
        subgraphs.append(subgraph)

    return subgraphs, labels

# Function to find and print the cut edges
def find_cut_edges(graph, labels):
    cut_edges = []
    for u, v, data in graph.edges(data=True):
        if labels[u] != labels[v]:
            cut_edges.append((u, v, data['weight']))
    return cut_edges






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


def build_from_EdgeAndFlag(edge_weights,edge_flag):
    G = nx.DiGraph()
    for (u,v) in edge_weights :
         if edge_flag[(u,v)]==1:
              G.add_edge(u,v,weight=edge_weights[(u,v)])
    return G

def build_from_GraphAndFlag (G,edge_flag):
    shrunkG = nx.DiGraph()
    for u, v, data in G.edges(data=True):
         if edge_flag[(u,v)]==1:
              shrunkG.add_edge(u,v,weight=data['weight'])
    return shrunkG

def build_from_subvertexset(shG,smallcom,edge_flag):
    smallG = nx.DiGraph()
    for u, v, data in shG.edges(data=True):
         if  u in smallcom and v in smallcom and edge_flag[(u,v)]==1:
              smallG.add_edge(u,v,weight=data['weight'])
    return smallG

def build_from_EdgeList(edge_weights):
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

def read_removed_edges(file_path,edge_flag):
    removed_weight=0
    with open(file_path,mode='r') as f:
        csv_reader = csv.reader(f)
        for row in csv_reader:
            source = int(row[0])
            dest = int(row[1])
            weight = int(row[2])  
            if edge_flag[(source,dest)]==1:
                edge_flag[(source,dest)]=0
                removed_weight+=weight
    return removed_weight


# Write the edge flag array
def write_removed_edges(output_file,edge_flag,edge_weights):
    with open(output_file, 'w') as f:
        for u,v in edge_flag:
            if edge_flag[(u,v)]==0:
                f.write(f"{u},{v},{edge_weights[(u,v)]}\n")

def read_config(file_path):
    with open(file_path,mode='r') as f:
        csv_reader = csv.reader(f)
        for row in csv_reader:
            fixcyclelen=int(row[0])
            mincyclelen=int(row[1])
            numcycles=int(row[2])
            maxcyclelen=int(row[3])
            subcomponentsize=int(row[4])
            heavysetsize=int(row[5])
    return fixcyclelen,mincyclelen,numcycles,maxcyclelen,subcomponentsize,heavysetsize

def read_time_limit(file_path):
    time_limit=0
    if os.path.exists(file_path):
        with open(file_path,mode='r') as f:
            csv_reader = csv.reader(f)
            for row in csv_reader:
                time_limit=int(row[0])
    return time_limit 

def write_config(fixcyclelen,mincyclelen,numcycles,maxcyclelen,subcomponentsize,heavysetsize,file_path):
    with open(file_path, 'w') as f:
            f.write(f"{fixcyclelen},{mincyclelen},{numcycles},{maxcyclelen},{subcomponentsize},{heavysetsize} \n")

num=0

#remove cycle in G and update edge_flag
def solve_ip_scc(G,edge_flag,SuperG):
    global num
    print(f"num of nodes is {G.number_of_nodes()}, num of edges is {G.number_of_edges()}\n")
    removed_weight=0
    model = gp.Model("min_feedback_arc_set")
    model.setParam('OutputFlag', 0)  # Silent mode

    # Create binary variables for each edge
    edge_vars = {}
    for u, v, data in G.edges(data=True):
        edge_vars[(u, v)] = model.addVar(vtype=GRB.BINARY, obj=data['weight'])

    cycles = nx.simple_cycles(G)
    for cycle in cycles:
        #print(f"the cycle is {cycle}")
        cycle_edges = [(cycle[i], cycle[(i+1) % len(cycle)]) for i in range(len(cycle))]
        model.addConstr(gp.quicksum(edge_vars[edge] for edge in cycle_edges) >= 1)

    # Optimize the model
    model.optimize()

    # Get the edges to be removed
    for edge, var in edge_vars.items():
            if var.x > 0.5:
               if edge_flag[(edge[0],edge[1])]==1:
                    removed_weight+=G[edge[0]][edge[1]]['weight']
                    edge_flag[(edge[0],edge[1])]=0
                    SuperG.remove_edge(edge[0],edge[1])
                    num+=1
    return removed_weight

# call OpenMP C function to remove cycles in subgraph of G built from given vertex set nodes
#maxlen,minlen are used to search circles with length in between
#num_long_cycles is used to search total number of cycles beyond the above cycles
#len_long_cycle is the largest long cycle size
#edge_flag will be used in search and updated based on the search results
def ompdfs_remove_cycle_edges(nodes,G, maxlen,minlen,num_long_cycles,len_long_cycle,time_limit,edge_flag):
    removed_weight=0
    global num

    commandline=f"rm -f tmp-omp-file.csv tmp-omp-removed-edges.csv"
    os.system(commandline)

    # Create the subgraph
    G_sub = G.subgraph(nodes).copy()

    with open("tmp-omp-file.csv", 'w') as f:
        for u, v, data in G_sub.edges(data=True):
            if edge_flag[(u,v)]==1:
               f.write(f"{u},{v},{data['weight']}\n")
    # search cycles in the subgraph
    commandline=f"./subompdfs tmp-omp-file.csv {maxlen} 0 {minlen} {num_long_cycles} {len_long_cycle} {time_limit}"
    if os.system(commandline)!=0 :
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        print("There is an error during call subompdfs")
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n")

    if os.path.exists("tmp-omp-removed-edges.csv"):
        with open("tmp-omp-removed-edges.csv",mode='r') as f:
            csv_reader = csv.reader(f)
            for row in csv_reader:
                source = int(row[0])
                dest = int(row[1])
                weight = int(row[2])  
                if edge_flag[(source,dest)]==1:
                    edge_flag[(source,dest)]=0
                    removed_weight+=weight
                    num+=1
    return removed_weight 

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
                    print(f"find cycle len is {len(cycle)} with vertices {cycle}")
                    for u, v in cycle_edges:
                        cycle_weights.append((u,v,edge_weights[(u,v)]))
                    min_edge = min(cycle_weights, key=lambda x: x[2])
                    removed_weight+=min_edge[2]
                    #removed_edges.add(min_edge)
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
    print(f"start from node {nodes[0]} and we have {len(nodes)} nodes")
    while len(nodes)>0: 
        node = nodes[0]
        #print(f"start from node {node} and we have {len(nodes)} nodes now")
        #print(f"enter dfs")
        dfs(node, [], rec_stack)
        if node in nodes:
            nodes.remove(node)


    return removed_weight 

def again_dfs_remove_cycle_edges(nodes,edge_weights,out_adj,edge_flag):
    removed_edges = set()
    def dfs(node, stack,  rec_stack):
        global num
        nonlocal removed_weight
        nonlocal tovisit
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
                    print(f"find cycle len is {len(cycle)} with vertices {cycle}")
                    for u, v in cycle_edges:
                        cycle_weights.append((u,v,edge_weights[(u,v)]))
                    min_edge = min(cycle_weights, key=lambda x: x[2])
                    removed_weight+=min_edge[2]
                    #removed_edges.add(min_edge)
                    print(f"removed {num+1} edges {min_edge}")
                    num=num+1
                    edge_flag[(min_edge[0],min_edge[1])]=0
            elif neighbor not in tovisit:
                    dfs(neighbor, stack, rec_stack)
                    if neighbor in tovisit:
                         tovisit.remove(neighbor)
        rec_stack.remove(node)
        restorenode=stack.pop()
        if node in tovisit:
                         tovisit.remove(node)
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
    return removed_weight 
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
                dfs(node, [], rec_stack)
            if node in tovisit:
                tovisit.remove(node)
        new_removed_weight = sum(w for u, v,w in removed_edges)
        removed_weight += new_removed_weight
        iternum=iternum+1
    print("finish  again bfs")
    return removed_weight 

# Normalize distributions to get probabilities
def normalize_distribution(distribution):
    total = sum(distribution.values())
    return {k: v / total for k, v in distribution.items()}

# Calculate min, max, and average values
def calculate_stats(distribution):
    keys = list(distribution.keys())
    values = list(distribution.values())
    min_val = min(keys)
    max_val = max(keys)
    mid_val= (min_val+max_val)/2
    val_4_3= max_val-(min_val+max_val)/4
    avg_val = sum(k * v for k, v in distribution.items())
    total_key  = sum(k  for k, v in distribution.items())
    above_mid = sum(k for k, v in distribution.items() if k>mid_val) 
    above_4_3 = sum(k for k, v in distribution.items() if k>val_4_3) 
    return min_val, max_val, avg_val/total_key, mid_val, above_mid, val_4_3, above_4_3

def select_heavy_node(dic,stats,percentage,heavyset):
     for node in dic:
        if dic[node]>stats[1] * percentage:
             heavyset.add(node)
     
def select_light_node(dic,stats,percentage,lightset):
     for node in dic:
        if dic[node]<stats[0] * percentage:
             lightset.add(node)

def calculate_light_set(G,percentage):
    # Calculate in-degree, out-degree, in-weight, and out-weight
    in_degrees = dict(G.in_degree())
    out_degrees = dict(G.out_degree())
    in_weights = {node: sum(data['weight'] for _, _, data in G.in_edges(node, data=True)) for node in G.nodes()}
    out_weights = {node: sum(data['weight'] for _, _, data in G.out_edges(node, data=True)) for node in G.nodes()}

    # Calculate the distributions
    in_degree_distribution = Counter(in_degrees.values())
    out_degree_distribution = Counter(out_degrees.values())
    in_weight_distribution = Counter(in_weights.values())
    out_weight_distribution = Counter(out_weights.values())

    in_degree_stats = calculate_stats(in_degree_distribution)
    out_degree_stats = calculate_stats(out_degree_distribution)
    in_weight_stats = calculate_stats(in_weight_distribution)
    out_weight_stats = calculate_stats(out_weight_distribution)

    lightset=set()
    select_light_node(in_degrees, in_degree_stats, percentage, lightset)
    select_light_node(out_degrees, out_degree_stats, percentage, lightset)
    #select_light_node(in_weights, in_weight_stats, percentage, lightset)
    #select_light_node(out_weights, out_weight_stats, percentage, lightset)
    return lightset

def calculate_heavy_set(G,percentage):
    # Calculate in-degree, out-degree, in-weight, and out-weight
    in_degrees = dict(G.in_degree())
    out_degrees = dict(G.out_degree())
    in_weights = {node: sum(data['weight'] for _, _, data in G.in_edges(node, data=True)) for node in G.nodes()}
    out_weights = {node: sum(data['weight'] for _, _, data in G.out_edges(node, data=True)) for node in G.nodes()}

    # Calculate the distributions
    in_degree_distribution = Counter(in_degrees.values())
    out_degree_distribution = Counter(out_degrees.values())
    in_weight_distribution = Counter(in_weights.values())
    out_weight_distribution = Counter(out_weights.values())

    in_degree_stats = calculate_stats(in_degree_distribution)
    out_degree_stats = calculate_stats(out_degree_distribution)
    in_weight_stats = calculate_stats(in_weight_distribution)
    out_weight_stats = calculate_stats(out_weight_distribution)

    heavyset=set()
    select_heavy_node(in_degrees, in_degree_stats, percentage, heavyset)
    select_heavy_node(out_degrees, out_degree_stats, percentage, heavyset)
    #select_heavy_node(in_weights, in_weight_stats, percentage, heavyset)
    #select_heavy_node(out_weights, out_weight_stats, percentage, heavyset)
    return heavyset




def remove_highdegree_lowweight_edge(G,p1,p2,edge_flag):
    # Calculate in-degree, out-degree, in-weight, and out-weight
    in_degrees = dict(G.in_degree())
    out_degrees = dict(G.out_degree())
    in_weights = {node: sum(data['weight'] for _, _, data in G.in_edges(node, data=True)) for node in G.nodes()}
    out_weights = {node: sum(data['weight'] for _, _, data in G.out_edges(node, data=True)) for node in G.nodes()}

    # Calculate the distributions
    in_degree_distribution = Counter(in_degrees.values())
    out_degree_distribution = Counter(out_degrees.values())
    in_weight_distribution = Counter(in_weights.values())
    out_weight_distribution = Counter(out_weights.values())

    in_degree_stats = calculate_stats(in_degree_distribution)
    out_degree_stats = calculate_stats(out_degree_distribution)
    in_weight_stats = calculate_stats(in_weight_distribution)
    out_weight_stats = calculate_stats(out_weight_distribution)
    diff_in=in_degree_stats[1]-in_degree_stats[0]
    diff_out=out_degree_stats[1]-out_degree_stats[0]

    num_removed=0
    for u,v,data in G.edges(data=True):
        if in_degrees[u] >=p1 and out_degrees[v]>=p1 and data['weight'] <3:
            edge_flag[(u,v)]=0
            num_removed+=1
            #print(f"({u},{v},{data['weight']} indegree={in_degrees[u]} max in degree {in_degree_stats[1]}, max out degree {out_degree_stats[1]} outdegree={out_degrees[v]},average in weight {in_weight_stats[2]} average out weight {out_weight_stats[2]}")
    return num_removed




def process_graph(file_path):
    print(f"read data")
    node_list, edge_weights, in_edges, out_edges = build_ArrayDataStructure(file_path)
    G=build_from_EdgeList(edge_weights)
    total=sum(edge_weights[(u,v)] for (u,v) in edge_weights)
    print(f"total number of nodes={len(node_list)}, total number of edges={len(edge_weights)}")
    print(f"sum of weight={total}")

    removed_weight=0
    removednum=0
    last_removed_weight=0
    edge_flag={(u,v):1 for (u,v) in edge_weights }

    #num_removed=remove_highdegree_lowweight_edge(G,100,100,edge_flag)
    #print(f"removed {num_removed} edges with degree >= 100")

    #removed_weight=read_removed_edges("removed.csv",edge_flag )
    for u,v in edge_flag:
        if edge_flag[(u,v)]==0:
           removednum+=1
           removed_weight+=edge_weights[(u,v)]
    print(f"to here removed {removednum} edges and the weight is {removed_weight}, percentage is {removed_weight/total*100}")

    shG=build_from_GraphAndFlag (G,edge_flag)




    subcomponentsize=5000
    fixcyclelen=5
    maxcyclelen=100
    numcycles=2000
    heavysetsize=200
    #reading from a file will allow us to change the value dynamically
    fixcyclelen, mincyclelen, numcycles,maxcyclelen,subcomponentsize,heavysetsize=read_config("subconfig.csv")

    numcheckacyclic=0
    while not nx.is_directed_acyclic_graph(shG):
        fixcyclelen, mincyclelen, numcycles,maxcyclelen,subcomponentsize,heavysetsize=read_config("subconfig.csv")
        time_limit=read_time_limit("time_limit.csv")
        print("the graph is not a DAG. Finding strongly connected components")
        scc=list(nx.strongly_connected_components(shG))
        print(f"number of scc is {len(scc)}")

        numcomponent=0
        oldnum=num
        numcheckacyclic+=1

        for component in scc:
            if len(component)==1:
                 continue

            numcomponent+=1
            print(f"{numcheckacyclic} check, handle the {numcomponent}th component with size {len(component)}")
            subnum=0


            if len(component)<1000:
                 G_sub = shG.subgraph(component).copy()
                 removed_weight1= solve_ip_scc(G_sub,edge_flag,shG)
                 removed_weight+=removed_weight1
                 print(f"The {numcomponent}th component, removed weight is {removed_weight1}, totally removed {removed_weight}, percentage is {removed_weight/total*100}\n")
                 continue


            totalsum1=0
            totalsum2=0
            totalsum0=0
            if  len(component) >1000:
                        G_sub = shG.subgraph(component).copy()
                        numpair,distance=read_num_pairs("numpair.csv")
                        percentage=0.8
                        heavyset =calculate_heavy_set(G_sub,percentage)
                        while len(heavyset)<2* numpair:
                            percentage-=0.1
                            heavyset =calculate_heavy_set(G_sub,percentage)
                        print("select high degree vertex set")
                        sum2=0
                        l=len(heavyset)
                        start_time=time.time()
                        for i in range(numpair):
                            if time.time()-start_time>time_limit:
                                  break
                            heavyset=list(heavyset)
                            target=heavyset[i]
                            source=heavyset[l-i-1]

                            if source!=target and nx.has_path(G_sub,source,target) and nx.has_path(G_sub,target,source):
                                      distance1=nx.shortest_path_length(G_sub,source=source,target=target)
                                      distance2=nx.shortest_path_length(G_sub,target=source,source=target)
                                      print(f"distance from {source} to {target} is {distance1}, from {target} to {source} is {distance2}") 
                                      cut_value1, cut_edges1 = find_minimum_cut(G_sub, source, target)
                                      cut_value2, cut_edges2 = find_minimum_cut(G_sub, target, source)
                                      print(f"value from {source} to {target} is {cut_value1}, from {target} to {source} is {cut_value2}") 
                                      weight1=0
                                      num1=0
                                      if cut_value1 < cut_value2:
                                          cut_edges=cut_edges1
                                      else:
                                          cut_edges=cut_edges2
                                      for u,v,w in cut_edges:
                                          edge_flag[(u,v)]=0
                                          G_sub.remove_edge(u,v)
                                          num1+=1
                                          weight1+=edge_weights[(u,v)]
                                      sum2+=weight1
                        removed_weight+=sum2
                        print(f"The {numcomponent}th component, removed weight is {sum2}, totally removed {removed_weight}, percentage is {removed_weight/total*100}\n")



            oldnum=removednum
            removednum=0
            actweight=0
            for u,v in edge_flag:
                if edge_flag[(u,v)]==0:
                    removednum+=1
                    actweight+=edge_weights[(u,v)]
            print(f"{numcheckacyclic} check,to here removed {removednum} edges and removed weight is {actweight} and percentage of remained  weight ={(total-actweight)/total *100}, removed weight is {removed_weight}")

            current_time = datetime.now()
            time_string = current_time.strftime("%Y%m%d_%H%M%S")
            output_file = f"{FileNameHead}-removed-edges-{time_string}.csv"
            print(f"removed edges weight increased {(actweight-last_removed_weight)/total}")
            if   (actweight-last_removed_weight)/total >0.05:
                        write_removed_edges(output_file,edge_flag,edge_weights) 
                        last_removed_weight=actweight
            else:
                 if numcheckacyclic % 100 ==0 and (actweight-last_removed_weight)/total >0.01:
                        write_removed_edges(output_file,edge_flag,edge_weights)
                        last_removed_weight=actweight


        print(f"sum of removed weight={actweight},percentage of remained  weight ={(total-actweight)/total *100}")

        print(f"build dag")
        shG=build_from_EdgeAndFlag(edge_weights,edge_flag)
        dag_weight=sum(data['weight'] for u, v, data in shG.edges(data=True))
        print(f"dag weight={dag_weight}")
        if dag_weight+removed_weight != total:
            print("accumulated weight in each step is not the same as the final removed weight")



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

