import pandas as pd
import networkx as nx
from networkx.algorithms.cycles import simple_cycles
import gurobipy as gp
from gurobipy import GRB
import sys
from datetime import datetime
import time

FileNameHead="DFS"
# Read the CSV file and create a directed graph
def read_graph_from_csv(file_path):
    #df = pd.read_csv(file_path)
    df=pd.read_csv(file_path, header=None, names=['source', 'destination', 'weight'])
    G = nx.DiGraph()
    for index, row in df.iterrows():
        G.add_edge(row['source'], row['destination'], weight=int(row['weight']))
        #print("source=",row['source'], "dest=",row['destination'], "weight=",row['weight'])
    return G

# Find all directed cycles in the graph
def find_cycles(G):
    return list(simple_cycles(G))
    #  next_n_cycles = list(itertools.islice(all_cycles, start, start + number))

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



# Calculate the total weight of the edges
def total_edge_weight(G):
    return sum(data['weight'] for u, v, data in G.edges(data=True))

# Calculate the weight of the removed edges
def removed_edge_weight(G, removed_edges):
    return sum(G[u][v]['weight'] for u, v in removed_edges)

# Identify the removed edges and construct a DAG
def construct_dag(G, removed_edges):
    G_dag = G.copy()
    G_dag.remove_edges_from(removed_edges)
    return G_dag

# Relabel the vertices in the DAG
def relabel_dag(G_dag):
    topological_order = list(nx.topological_sort(G_dag))
    mapping = {node: i for i, node in enumerate(topological_order)}
    #G_dag = nx.relabel_nodes(G_dag, mapping)
    #return G_dag, mapping
    return mapping

# Write the original vertex ID and its relative order to a file
def write_relabelled_nodes_to_file(mapping, output_file):
    with open(output_file, 'w') as f:
        for node, order in mapping.items():
            f.write(f"{node},{order}\n")

def dfs_remove_cycle_edges(graph):
    removed_edges = set()
    #removed_edges =[]
    def dfs(node, stack, visited, rec_stack):
        visited.add(node)
        rec_stack.add(node)
        stack.append(node)
        for neighbor, weight in graph[node].items():
            if neighbor not in visited:
                dfs(neighbor, stack, visited, rec_stack)
            elif neighbor in rec_stack:
                cycle = stack[stack.index(neighbor):] + [neighbor]
                #min_edge = min(((u, v) for u, v in zip(cycle, cycle[1:] + [cycle[0]])), key=lambda edge: graph[edge[0]][edge[1]]['weight'])
                #removed_edges.add(min_edge)

                cycle_edges = [(cycle[i], cycle[i + 1]) for i in range(len(cycle) - 1)]
                cycle_weights = [(u, v, graph[u][v]['weight']) for u, v in cycle_edges]
                min_edge = min(cycle_weights, key=lambda x: x[2])
                removed_edges.add(min_edge)
                print("find cycle edges",cycle_edges)
                print("remove edge",min_edge)
                #removed_edges.append(min_edge)
                #removed_edges.append([min_edge[0],min_edge[1]])
        stack.pop()
        rec_stack.remove(node)
        return False
    
    visited = set()
    rec_stack = set()
    for node in graph.nodes():
        if node not in visited:
            dfs(node, [], visited, rec_stack)
    #removed_weight = sum(graph[u][v]['weight'] for u, v in removed_edges)
    removed_weight = sum(w for u, v,w in removed_edges)
    graph.remove_edges_from([(u, v) for u, v, w in removed_edges])
    return removed_weight, removed_edges


def dfs_remove_cycle_edges_2(graph):
    removed_edges = []
    total_weight = sum(data['weight'] for u, v, data in graph.edges(data=True))

    def dfs(node, stack, visited, rec_stack):
        visited.add(node)
        rec_stack.add(node)
        stack.append(node)
        for neighbor in graph[node]:
            if neighbor not in visited:
                if dfs(neighbor, stack, visited, rec_stack):
                    return True
            elif neighbor in rec_stack:
                cycle = stack[stack.index(neighbor):] + [neighbor]
                cycle_edges = [(cycle[i], cycle[i + 1]) for i in range(len(cycle) - 1)]
                cycle_weights = [(u, v, graph[u][v]['weight']) for u, v in cycle_edges]
                min_edge = min(cycle_weights, key=lambda x: x[2])
                removed_edges.append(min_edge)
                return True
        stack.pop()
        rec_stack.remove(node)
        return False

    visited = set()
    rec_stack = set()
    for node in graph.nodes():
        if node not in visited:
            dfs(node, [], visited, rec_stack)
    
    removed_weight = sum(w for u, v, w in removed_edges)
    
    # Create a new graph by removing the edges in removed_edges
    new_graph = graph.copy()
    new_graph.remove_edges_from([(u, v) for u, v, w in removed_edges])
    
    return total_weight, removed_weight, removed_edges, new_graph



def dfs_remove_cycle_edges_3(graph):
    removed_edges = []
    total_weight = sum(data['weight'] for u, v, data in graph.edges(data=True))

    def iterative_dfs(start_node):
        stack = [(start_node, iter(graph[start_node]))]

        while stack:
            node, neighbors = stack[-1]
            if node not in visited:
                visited.add(node)
                rec_stack.add(node)
                path.append(node)

            try:
                neighbor = next(neighbors)
                if neighbor not in visited:
                    stack.append((neighbor, iter(graph[neighbor])))
                elif neighbor in rec_stack:
                    cycle = path[path.index(neighbor):] + [neighbor]
                    cycle_edges = [(cycle[i], cycle[i + 1]) for i in range(len(cycle) - 1)]
                    cycle_weights = [(u, v, graph[u][v]['weight']) for u, v in cycle_edges]
                    min_edge = min(cycle_weights, key=lambda x: x[2])
                    removed_edges.append(min_edge)
            except StopIteration:
                stack.pop()
                rec_stack.remove(node)
                path.pop()

    visited = set()
    rec_stack = set()
    path = []
    for node in graph.nodes():
        if node not in visited:
            iterative_dfs(node)

    removed_weight = sum(w for u, v, w in removed_edges)
    graph.remove_edges_from([(u, v) for u, v, w in removed_edges])
    return total_weight, removed_weight, removed_edges,dag_G




def remove_duplicate_edges_and_self_loops(graph):
    # Create a dictionary to store the summed weights of edges
    edge_weights = {}

    # Iterate through all edges in the graph
    for u, v, data in graph.edges(data=True):
        if u == v:
            # Skip self-loops
            continue
        if (u, v) in edge_weights:
            edge_weights[(u, v)] += data['weight']
        else:
            edge_weights[(u, v)] = data['weight']

    # Create a new graph and add the summed edges
    new_graph = nx.DiGraph()
    for (u, v), weight in edge_weights.items():
        new_graph.add_edge(u, v, weight=weight)

    return new_graph


# Main function to perform all steps
def process_graph(file_path):

    starttime = time.time()
    dupG = read_graph_from_csv(file_path)
    endtime = time.time()
    executiontime=endtime-starttime
    current_time = datetime.now()
    time_string = current_time.strftime("%Y%m%d_%H%M%S")
    # Create the file name using the formatted time string
    oneoutputfile = f"{FileNameHead}-{time_string}.txt"


    with open(oneoutputfile, 'w') as f:
            f.write(f"reading file {file_path} takes {executiontime} seconds \n\n")


    total_weight=total_edge_weight(dupG)
    #print("merge multiple edges and remove self-loops")
    starttime = time.time()
    G = remove_duplicate_edges_and_self_loops(dupG)
    endtime = time.time()
    executiontime=endtime-starttime
    with open(oneoutputfile, 'a') as f:
            f.write(f"Remove Duplicated edges in  file {file_path} takes {executiontime} seconds \n\n")

    starttime = time.time()
    removed_weight, removed_edges =dfs_remove_cycle_edges(G)
    endtime = time.time()
    executiontime=endtime-starttime
    current_time = datetime.now()
    # Create the file name using the formatted time string
    total_weight = total_edge_weight(dupG)
    percentage_removed = (removed_weight / total_weight) * 100
    with open(oneoutputfile, 'a') as f:
        f.write(f"DFS find all removed edges in file {file_path} uses {executiontime} seconds \n")
        f.write(f"Total Edge Weight:{total_weight}, Total number of Edges= {G.number_of_edges()}\n")
        f.write(f"Removed Edge Weight: {removed_weight} Removed number of Edges={len(removed_edges)}\n")
        f.write(f"Percentage of Removed Edges Weight: {percentage_removed}\n")
        f.write(f"All removed edges are as follows \n")
        for u, v ,w in removed_edges:
            f.write(f"{u},{v},{w}\n")

    if not nx.is_directed_acyclic_graph(G):

        print("the graph is not a DAG. There is something wrong")
        print(find_cycles(G))
        exit(0)
    endtime = time.time()
    executiontime=endtime-starttime
    current_time = datetime.now()
    with open(oneoutputfile, 'a') as f:
            f.write(f"\n construct DAG graph in  file {file_path} uses {executiontime} seconds \n\n")

    starttime = time.time()
    #G_dag_relabelled, mapping = relabel_dag(G_dag)
    mapping = relabel_dag(G)
    dag_weight=total_edge_weight(G)
    endtime = time.time()
    executiontime=endtime-starttime
    current_time = datetime.now()
    with open(oneoutputfile, 'a') as f:
            f.write(f"Topological sort in  generated DAG graph of  file {file_path} uses {executiontime} seconds \n")
            f.write(f"The DAG graph weight percentage is {dag_weight/total_weight * 100} \n\n")




    starttime = time.time()
    current_time = datetime.now()
    time_string = current_time.strftime("%Y%m%d_%H%M%S")
    #Create the file name using the formatted time string
    output_file = f"{FileNameHead}-07-Relabel-{time_string}.csv"
    write_relabelled_nodes_to_file(mapping, output_file)
    endtime = time.time()
    executiontime=endtime-starttime
    current_time = datetime.now()
    with open(oneoutputfile, 'a') as f:
            f.write(f"write label of file {file_path} uses {executiontime} seconds \n\n")

file_path = sys.argv[1]


sys.setrecursionlimit(900000)
process_graph(file_path)



