import pandas as pd
import networkx as nx
from networkx.algorithms.cycles import simple_cycles
import gurobipy as gp
from gurobipy import GRB
import sys

# Step 1: Read the CSV file and create a directed graph
def read_graph_from_csv(file_path):
    #df = pd.read_csv(file_path)
    df=pd.read_csv(file_path, header=None, names=['source', 'destination', 'weight'])
    G = nx.DiGraph()
    for index, row in df.iterrows():
        G.add_edge(row['source'], row['destination'], weight=row['weight'])
    return G

# Step 2: Find all directed cycles in the graph
def find_cycles(G, start=0,number=-1):
    return list(simple_cycles(G))
    all_cycles = nx.simple_cycles(G)
    if number==-1:
       return all_cycles
    else :
       next_n_cycles = list(itertools.islice(all_cycles, start, start + number))
       return next_n_cycles

# Step 3: Formulate the integer programming problem
def solve_ip_for_minimum_feedback_arc_set(G, cycles):
    model = gp.Model("min_feedback_arc_set")
    model.setParam('OutputFlag', 0)  # Silent mode

    # Create binary variables for each edge
    edge_vars = {}
    for u, v, data in G.edges(data=True):
        edge_vars[(u, v)] = model.addVar(vtype=GRB.BINARY, obj=data['weight'])

    # Add constraints for each cycle
    for cycle in cycles:
        cycle_edges = [(cycle[i], cycle[(i+1) % len(cycle)]) for i in range(len(cycle))]
        model.addConstr(gp.quicksum(edge_vars[edge] for edge in cycle_edges) >= 1)

    # Optimize the model
    model.optimize()

    # Get the edges to be removed
    removed_edges = [edge for edge, var in edge_vars.items() if var.x > 0.5]
    return removed_edges


# Calculate the total weight of the edges
def total_edge_weight(G):
    return sum(data['weight'] for u, v, data in G.edges(data=True))

# Calculate the weight of the removed edges
def removed_edge_weight(G, removed_edges):
    return sum(G[u][v]['weight'] for u, v in removed_edges)

# Step 4: Identify the removed edges and construct a DAG
def construct_dag(G, removed_edges):
    G_dag = G.copy()
    G_dag.remove_edges_from(removed_edges)
    return G_dag

# Step 5: Relabel the vertices in the DAG
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

# Main function to perform all steps
def process_graph(file_path, output_file):
    G = read_graph_from_csv(file_path)
    cycles = find_cycles(G,0,5)
    removed_edges = solve_ip_for_minimum_feedback_arc_set(G, cycles)
    total_weight = total_edge_weight(G)
    removed_weight = removed_edge_weight(G, removed_edges)
    percentage_removed = (removed_weight / total_weight) * 100

    #print("Removed Edges:", removed_edges)
    print("Total Edge Weight:", total_weight)
    print("Removed Edge Weight:", removed_weight)
    print("Percentage of Removed Edges Weight:", percentage_removed)

    G_dag = construct_dag(G, removed_edges)
    G_dag_relabelled, mapping = relabel_dag(G_dag)

    write_relabelled_nodes_to_file(mapping, output_file)

# Example usage
file_path = sys.argv[1]
output_file = 'relabelled_nodes.csv'
process_graph(file_path, output_file)


