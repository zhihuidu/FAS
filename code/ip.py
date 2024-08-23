import networkx as nx
import csv

# Reading the CSV file and creating the directed graph
def read_graph_from_csv(file_path):
    G = nx.DiGraph()
    with open(file_path, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            source, destination, weight = int(row[0]), int(row[1]), float(row[2])
            G.add_edge(source, destination, weight=weight)
    return G

# Example usage
file_path = 'graph.csv'
G = read_graph_from_csv(file_path)


cycles = list(nx.simple_cycles(G))
print("All directed cycles:", cycles)



import pandas as pd
import networkx as nx
from itertools import combinations
from networkx.algorithms.cycles import simple_cycles
import gurobipy as gp
from gurobipy import GRB

# Step 1: Read the CSV file and create a directed graph
def read_graph_from_csv(file_path):
    df = pd.read_csv(file_path)
    G = nx.DiGraph()
    for index, row in df.iterrows():
        G.add_edge(row['source'], row['destination'], weight=row['weight'])
    return G

# Step 2: Find all directed cycles in the graph
def find_cycles(G):
    return list(simple_cycles(G))

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

# Step 4: Identify the removed edges and construct a DAG
def construct_dag(G, removed_edges):
    G_dag = G.copy()
    G_dag.remove_edges_from(removed_edges)
    return G_dag

# Step 5: Relabel the vertices in the DAG
def relabel_dag(G_dag):
    mapping = {node: i for i, node in enumerate(sorted(G_dag.nodes))}
    G_dag = nx.relabel_nodes(G_dag, mapping)
    return G_dag

# Main function to perform all steps
def process_graph(file_path):
    G = read_graph_from_csv(file_path)
    cycles = find_cycles(G)
    removed_edges = solve_ip_for_minimum_feedback_arc_set(G, cycles)
    G_dag = construct_dag(G, removed_edges)
    G_dag_relabelled = relabel_dag(G_dag)

    print("Removed Edges:", removed_edges)
    print("DAG Nodes:", G_dag_relabelled.nodes())
    print("DAG Edges:", G_dag_relabelled.edges(data=True))

# Example usage
file_path = 'graph.csv'
process_graph(file_path)

