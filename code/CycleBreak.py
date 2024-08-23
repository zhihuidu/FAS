import pandas as pd
import networkx as nx
from networkx.algorithms.cycles import simple_cycles
import gurobipy as gp
from gurobipy import GRB
import sys
from datetime import datetime
import time

FileNameHead="CycleBreak"
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
    G_dag = nx.relabel_nodes(G_dag, mapping)
    return G_dag, mapping

# Write the original vertex ID and its relative order to a file
def write_relabelled_nodes_to_file(mapping, output_file):
    with open(output_file, 'w') as f:
        for node, order in mapping.items():
            f.write(f"{node},{order}\n")

def greedy_feedback_arc_set(graph):
    def find_cycle(graph):
        try:
            cycle = nx.find_cycle(graph, orientation='original')
            return cycle
        except nx.NetworkXNoCycle:
            return None

    removed_edges = []
    while True:
        cycle = find_cycle(graph)
        if cycle is None:
            break
        min_edge = min(cycle, key=lambda edge: graph[edge[0]][edge[1]]['weight'])
        graph.remove_edge(*min_edge[:2])
        removed_edges.append([min_edge[0],min_edge[1]])
    
    return removed_edges


# Main function to perform all steps
def process_graph(file_path):

    starttime = time.time()
    G = read_graph_from_csv(file_path)
    endtime = time.time()
    executiontime=endtime-starttime
    current_time = datetime.now()
    time_string = current_time.strftime("%Y%m%d_%H%M%S")
    # Create the file name using the formatted time string
    oneoutput_file = f"{FileNameHead}-{time_string}.txt"
    with open(oneoutput_file, 'w') as f:
            f.write(f"reading file {file_path} takes {executiontime} seconds \n")

    starttime = time.time()
    oldG=G.copy()
    removed_edges = greedy_feedback_arc_set(G)
    endtime = time.time()
    executiontime=endtime-starttime
    total_weight = total_edge_weight(oldG)
    removed_weight = removed_edge_weight(oldG, removed_edges)
    percentage_removed = (removed_weight / total_weight) * 100
    print("Total Edge Weight:", total_weight," Total number of Edges=",oldG.number_of_edges())
    print("Removed Edge Weight:", removed_weight, " Removed number of Edges=", len(removed_edges))
    print("Percentage of Removed Edges Weight:", percentage_removed)
    output_file = f"{FileNameHead}-04-{file_path}-ipsolver-{time_string}.txt"
    with open(oneoutput_file, 'a') as f:
        f.write(f"Using Cycle break to find all removed edges in cycle graph of file {file_path} uses {executiontime} seconds \n")
        f.write(f"Total Edge Weight:{total_weight}, Total number of Edges= {oldG.number_of_edges()}\n")
        f.write(f"Removed Edge Weight: {removed_weight} Removed number of Edges={len(removed_edges)}\n")
        f.write(f"Percentage of Removed Edges Weight: {percentage_removed}\n")
        f.write(f"All removed edges are as follows \n")
        for u, v in removed_edges:
            f.write(f"{u},{v},{oldG[u][v]['weight']}\n")


    starttime = time.time()
    G_dag = G
    G_dag_relabelled, mapping = relabel_dag(G_dag)
    endtime = time.time()
    executiontime=endtime-starttime
    current_time = datetime.now()
    with open(oneoutput_file, 'a') as f:
            f.write(f"Topological sort in  generated DAG graph of  file {file_path} uses {executiontime} seconds \n")


    starttime = time.time()
    current_time = datetime.now()
    time_string = current_time.strftime("%Y%m%d_%H%M%S")
    #Create the file name using the formatted time string
    output_file = f"{FileNameHead}-07-{file_path}-relabel-{time_string}.csv"
    write_relabelled_nodes_to_file(mapping, output_file)
    endtime = time.time()
    executiontime=endtime-starttime
    with open(oneoutput_file, 'a') as f:
            f.write(f"write label of file {file_path} uses {executiontime} seconds \n")

file_path = sys.argv[1]
process_graph(file_path)


