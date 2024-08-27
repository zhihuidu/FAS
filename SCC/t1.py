import networkx as nx
import matplotlib.pyplot as plt
from collections import Counter

import pandas as pd
from networkx.algorithms.cycles import simple_cycles
import gurobipy as gp
from gurobipy import GRB
import sys
from datetime import datetime
import time
import csv
import random
from itertools import permutations


FileNameHead="Clean-SCC-IP"

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


# Find all directed triangle edges in the graph
def find_triangle_edges(G):
    triangle_edges=set()
    for u, v in G.edges():
        # Check for a common neighbor that completes a triangle
        for w in G.neighbors(v):
            if G.has_edge(w, u) :
                if u==min(u,v,w):
                    triangle_edges.add((u,v,w))
                else:
                    if v==min(u,v,w):
                         triangle_edges.add((v,w,u))
                    else:
                         triangle_edges.add((w,u,v))

    return triangle_edges

# Find all directed quadrilateral  edges in the graph
def find_quadrilateral_edges(G):
    quadrilateral_edges=set()
    for u, v in G.edges():
        # Check for a common neighbor that completes a triangle
        for w in G.neighbors(v):
            for x in G.neighbors(w):
                if G.has_edge(x, u):
                    if u==min(u,v,w,x):
                       quadrilateral_edges.add((u,v,w,x))
                    else:
                       if v==min(u,v,w,x):
                         quadrilateral_edges.add((v,w,x,u))
                       else:
                          if w==min(u,v,w,x):
                             quadrilateral_edges.add((w,x,u,v))
                          else:
                             quadrilateral_edges.add((x,u,v,w))

    return quadrilateral_edges

def build_graph_from_component(G,component):
    com_graph = nx.DiGraph()
    
    # Add edges from triangle set to the new graph
    for u in component:
            for v in G.neighbors(u):
                if v in component:
                    com_graph.add_edge(u,v,weight=G[u][v]['weight'])
    return com_graph

def build_triangle_graph(original_graph,edgeset):
    tri_graph = nx.DiGraph()
    
    # Add edges from triangle set to the new graph
    for u,v,w in edgeset:
                weight = original_graph[u][v]['weight']
                tri_graph.add_edge(u, v, weight=weight)
                weight = original_graph[v][w]['weight']
                tri_graph.add_edge(v, w, weight=weight)
                weight = original_graph[w][u]['weight']
                tri_graph.add_edge(w, u, weight=weight)

    return tri_graph

def build_quadrilateral_graph(original_graph,edgeset):
    quad_graph = nx.DiGraph()
    
    # Add edges from triangle set to the new graph
    for u,v,w,x in edgeset:
                weight = original_graph[u][v]['weight']
                quad_graph.add_edge(u, v, weight=weight)
                weight = original_graph[v][w]['weight']
                quad_graph.add_edge(v, w, weight=weight)
                weight = original_graph[w][x]['weight']
                quad_graph.add_edge(w, x, weight=weight)
                weight = original_graph[x][u]['weight']
                quad_graph.add_edge(x, u, weight=weight)

    return quad_graph
def build_cycle_graph(original_graph):
    """
    Build a new directed graph based on cycles from the original directed graph.
    
    Parameters:
    original_graph (networkx.DiGraph): The original directed graph.
    
    Returns:
    networkx.DiGraph: A new directed graph containing edges from cycles.
    """
    cycle_graph = nx.DiGraph()
    
    # Get all cycles in the original graph
    cycles = nx.simple_cycles(original_graph)

    # Add edges from each cycle to the new graph
    for cycle in cycles:
        # Add edges of the cycle to the new graph
        for i in range(len(cycle)):
            source = cycle[i]
            destination = cycle[(i + 1) % len(cycle)]  # wrap around to the start
            # Check if the edge exists in the original graph
            if original_graph.has_edge(source, destination):
                weight = original_graph[source][destination]['weight']
                cycle_graph.add_edge(source, destination, weight=weight)

    return cycle_graph


# Formulate the integer programming problem
def solve_ip_scc(G,edge_flag,filename):
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
    with open(filename, 'a') as f:
        for edge, var in edge_vars.items():
            if var.x > 0.5:
               edge_flag[(edge[0],edge[1])]=0
               print(f"remove edge ({edge[0]},{edge[1]})")
               f.write(f"remove edge ({edge[0]},{edge[1]})\n")

    #removed_edges = [edge for edge, var in edge_vars.items() if var.x > 0.5]
    #print(f"removed edges={removed_edges}")
    return 0





def remove_small_cycles(nodes,edge_weights,out_adj,edge_flag):
    print(f"clean the size 2,3,4 and 5 loops")
    global num
    for node in nodes:
        for neighbor, weight in out_adj[node]:
            if neighbor < node or edge_flag[(node,neighbor)]==0:
                  continue
            if (neighbor,node) in edge_flag:
                 if edge_flag[(neighbor,node)]==1:
                        if weight < edge_weights[(neighbor,node)]:
                              edge_flag[(node,neighbor)]=0
                        else:
                              edge_flag[(neighbor,node)]=0
                              print(f"removed {num} edges ({neighbor},{node})")
                        num+=1
            for third, weight1 in out_adj[neighbor]:
                if third < node or edge_flag[(neighbor,third)]==0:
                        continue
                if (third,node) in edge_flag:
                      if edge_flag[(third,node)]==1:
                          weight2=edge_weights[(third,node)]
                          if weight==min(weight,weight1,weight2):
                               edge_flag[(node,neighbor)]=0
                               print(f"removed {num} edges ({node},{neighbor})")
                               num+=1
                          else:
                              if weight1==min(weight,weight1,weight2):
                                   edge_flag[(neighbor,third)]=0
                                   print(f"removed {num} edges ({neighbor},{third})")
                                   num+=1
                              else:
                                   edge_flag[(third,node)]=0
                                   print(f"removed {num} edges ({third},{node})")
                                   num+=1
                for four,weight2 in out_adj[third]:
                         if four < node  or edge_flag[(third,four)]==0:
                                continue
                         if (four,node) in edge_flag:
                             if edge_flag[(four,node)]==1:
                                 weight3=edge_weights[(four,node)]
                                 if weight==min(weight,weight1,weight2,weight3):
                                      edge_flag[(node,neighbor)]=0
                                      print(f"removed {num} edges ({node},{neighbor})")
                                      num+=1
                                 else:
                                      if weight1==min(weight,weight1,weight2,weight3):
                                           edge_flag[(neighbor,third)]=0
                                           print(f"removed {num} edges ({node},{neighbor})")
                                           num+=1
                                      else:
                                          if weight2==min(weight,weight1,weight2,weight3):
                                                edge_flag[(third,four)]=0
                                                print(f"removed {num} edges ({third},{four})")
                                                num+=1
                                          else:
                                                edge_flag[(four,node)]=0
                                                print(f"removed {num} edges ({four},{node})")
                                                num+=1


                         for five, weight3 in out_adj[four]:
                               if five < node  or edge_flag[(four,five)]==0:
                                      continue
                               if (five,node) in edge_flag:
                                   if edge_flag[(five,node)]==1:
                                       weight4=edge_weights[(five,node)]
                                       if weight==min(weight,weight1,weight2,weight3,weight4):
                                            edge_flag[(node,neighbor)]=0
                                            print(f"removed {num} edges ({node},{neighbor})")
                                            num+=1
                                       else:
                                            if weight1==min(weight,weight1,weight2,weight3,weight4):
                                                 edge_flag[(neighbor,third)]=0
                                                 print(f"removed {num} edges ({node},{neighbor})")
                                                 num+=1
                                            else:
                                                if weight2==min(weight,weight1,weight2,weight3,weight4):
                                                      edge_flag[(third,four)]=0
                                                      print(f"removed {num} edges ({third},{four})")
                                                      num+=1
                                                else:
                                                      if weight3==min(weight,weight1,weight2,weight3,weight4):
                                                            edge_flag[(four,five)]=0
                                                            print(f"removed {num} edges ({four},{node})")
                                                            num+=1
                                                      else:
                                                            edge_flag[(five,node)]=0
                                                            print(f"removed {num} edges ({five},{node})")
                                                            num+=1


    print(f"after clean the size 2,3,4 and 5 loops")


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







# Formulate the integer programming problem
def solve_ip_for_minimum_feedback_arc_set(G, cycles):
    model = gp.Model("min_feedback_arc_set")
    model.setParam('OutputFlag', 0)  # Silent mode

    # Create binary variables for each edge
    edge_vars = {}
    for u, v, data in G.edges(data=True):
        edge_vars[(u, v)] = model.addVar(vtype=GRB.BINARY, obj=data['weight'])

    # Add constraints for each cycle
    for cycle in cycles:
        #print(f"the cycle is {cycle}")
        cycle_edges = [(cycle[i], cycle[(i+1) % len(cycle)]) for i in range(len(cycle))]
        model.addConstr(gp.quicksum(edge_vars[edge] for edge in cycle_edges) >= 1)

    # Optimize the model
    model.optimize()

    # Get the edges to be removed
    removed_edges = [edge for edge, var in edge_vars.items() if var.x > 0.5]
    return removed_edges


# Main function to perform all steps

file_path = sys.argv[1]
print(f"read data")
node_list, edge_weights, in_edges, out_edges = create_adjacency_lists(file_path)
total=sum(edge_weights[(u,v)] for (u,v) in edge_weights)
print(f"total number of nodes={len(node_list)}, total number of edges={len(edge_weights)}")
print(f"sum of weight={total}")

removed_weight=0
G=build_graph(edge_weights)
edge_flag={(u,v):1 for (u,v) in edge_weights }





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

in_degree_stats = calculate_stats(in_degree_distribution)
out_degree_stats = calculate_stats(out_degree_distribution)
in_weight_stats = calculate_stats(in_weight_distribution)
out_weight_stats = calculate_stats(out_weight_distribution)


percentage=0.96
def select_node(dic,stats,percentage,heavyset):
     for node in dic:
        if dic[node]>stats[1] * percentage:
             heavyset.add(node)
     

heavyset=set()
while len(heavyset) <1000:
    select_node(in_degrees, in_degree_stats, percentage, heavyset)
    select_node(out_degrees, out_degree_stats, percentage, heavyset)
    select_node(in_weights, in_weight_stats, percentage, heavyset)
    select_node(out_weights, out_weight_stats, percentage, heavyset)
    percentage-=0.05
    print(f"size of the heavy set is {len(heavyset)}, percentage is {percentage}")

print(f"size of the heavy set is {len(heavyset)}")
exit(0)
#in_degree_distribution = normalize_distribution(in_degree_distribution)
#out_degree_distribution = normalize_distribution(out_degree_distribution)
#in_weight_distribution = normalize_distribution(in_weight_distribution)
#out_weight_distribution = normalize_distribution(out_weight_distribution)
# Save distributions and statistics to a text file
with open('distributions.txt', 'w') as f:
    def write_distribution_stats(f, name, distribution, stats):
        f.write(f"{name} Distribution:\n")
        for k, v in distribution.items():
            #f.write(f"Value: {k}, Probability: {v}\n")
            f.write(f"Key : {k}, Num: {v}\n")
        f.write(f"Min: {stats[0]}, Max: {stats[1]}, Average: {stats[2]:.4f} Mid Key:{stats[3]:.2f}, Num :{stats[4]} 3/4 Key:{stats[5]}, Num:{stats[6]}\n\n")
    
    write_distribution_stats(f, "In-Degree", in_degree_distribution, in_degree_stats)
    write_distribution_stats(f, "Out-Degree", out_degree_distribution, out_degree_stats)
    write_distribution_stats(f, "In-Weight", in_weight_distribution, in_weight_stats)
    write_distribution_stats(f, "Out-Weight", out_weight_distribution, out_weight_stats)

# Plotting each distribution in a separate figure
def plot_and_save_distribution(distribution, title, xlabel, ylabel, filename):
    plt.figure(figsize=(6, 5))
    plt.bar(distribution.keys(), distribution.values(), width=0.5, color='skyblue')
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    
    # Set x-axis limits from the smallest to the largest values
    #plt.xlim(min(distribution.keys()) - 0.5, max(distribution.keys()) + 0.5)
    #plt.xlim(min(distribution.keys()) , max(distribution.keys()) *1.1)
    plt.xlim(min(filter(lambda v: v > 0, distribution.keys())), max(distribution.keys()) * 1.1)
    
    # Set y-axis to log base 2 and ensure it covers all data points
    plt.yscale('log', base=2)
    plt.xscale('log', base=2)
    plt.ylim(min(filter(lambda v: v > 0, distribution.values())), max(distribution.values()) * 1.1)
    
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

#plot_and_save_distribution(in_degree_distribution, "In-Degree Distribution", "In-Degree", "Probability", "in_degree_distribution.png")
plot_and_save_distribution(in_degree_distribution, "In-Degree Distribution", "In-Degree", "Number", "ori_in_degree_distribution.png")
#plot_and_save_distribution(out_degree_distribution, "Out-Degree Distribution", "Out-Degree", "Probability", "out_degree_distribution.png")
plot_and_save_distribution(out_degree_distribution, "Out-Degree Distribution", "Out-Degree", "Number", "ori_out_degree_distribution.png")
#plot_and_save_distribution(in_weight_distribution, "In-Weight Distribution", "In-Weight", "Probability", "in_weight_distribution.png")
plot_and_save_distribution(in_weight_distribution, "In-Weight Distribution", "In-Weight", "Number", "ori_in_weight_distribution.png")
#plot_and_save_distribution(out_weight_distribution, "Out-Weight Distribution", "Out-Weight", "Probability", "out_weight_distribution.png")
plot_and_save_distribution(out_weight_distribution, "Out-Weight Distribution", "Out-Weight", "Number", "ori_out_weight_distribution.png")

