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


def build_small_graph(shG,subcom,edge_flag):
    smallG = nx.DiGraph()
    for u, v, data in shG.edges(data=True):
         if edge_flag[(u,v)]==1 and u in subcom and v in subcom:
              smallG.add_edge(u,v,weight=data['weight'])
    return smallG
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

# Calculate the weight of the removed edges
def removed_edge_weight(G, edge_flag):
    removed_w=0
    for u, v in edge_flag:
        if edge_flag[(u,v)]==0:
            removed_w += G[u][v]['weight']
    return removed_w

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

def process_graph(file_path):
    print(f"read data")
    node_list, edge_weights, in_edges, out_edges = create_adjacency_lists(file_path)
    total=sum(edge_weights[(u,v)] for (u,v) in edge_weights)
    print(f"total number of nodes={len(node_list)}, total number of edges={len(edge_weights)}")
    print(f"sum of weight={total}")

    removed_weight=0
    G=build_graph(edge_weights)
    edge_flag={(u,v):1 for (u,v) in edge_weights }
    removed_weight = again_dfs_remove_cycle_edges(node_list, edge_weights,out_edges,edge_flag )

    remove_small_cycles(node_list,edge_weights,out_edges,edge_flag)
    current_time = datetime.now()
    time_string = current_time.strftime("%Y%m%d_%H%M%S")
    output_file=FileNameHead+"removed_edge"+time_string+".csv"
    with open(output_file, 'w') as f:
        for u, v in edge_flag:
            if edge_flag[(u,v)]==0:
                 f.write(f"{u},{v}\n")
    print(f"Totally removed weight is {removed_weight} by reading removed edges\n")


    shG=build_new_shrunk_graph (G,edge_flag)
    current_time = datetime.now()
    time_string = current_time.strftime("%Y%m%d_%H%M%S")
    output_file = f"{FileNameHead}-step-{time_string}.csv"
    while not nx.is_directed_acyclic_graph(shG):
        print("the graph is not a DAG.")
        print(f"strongly connected components")
        scc=list(nx.strongly_connected_components(shG))
        numcom=0
        oldnum=num
        newsize=2000
        for component in scc:
            if len(component)==1:
                 continue
            print(f"handle the {numcom}th component with {len(component)}")
            numcom += 1
            print(f"handle component {numcom}th")
            while len(component)>newsize:
                randnum=0
                subcom=random.sample(list(component),newsize)
                print(f"handle {random}th random sub component of {numcom}th component with {len(component)}\n")
                randnum+=1
                component.difference_update(subcom)
                smallG=build_small_graph(shG,subcom,edge_flag)
                solve_ip_scc(smallG,edge_flag)
                with open(output_file, 'a') as f:
                    f.write(f"handle {random}th random sub component of {numcom}th component with {len(component)}\n")
                if num-oldnum<10:
                    newsize=min(newsize*2,component)
                oldnum=num

            comG=build_graph_from_component(G,component)
            solve_ip_scc(comG,edge_flag)
            removed_weight = removed_edge_weight(G, edge_flag)
            print(f"Totally removed weight is {removed_weight}, percentage is {removed_weight/total*100}\n\n")

        print(f"sum of removed weight={removed_weight},percentage of remained  weight ={(total-removed_weight)/total *100}")

        print(f"build dag")
        shG=build_new_shrunk_graph (G,edge_flag)
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

