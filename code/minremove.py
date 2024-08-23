import pandas as pd
import networkx as nx
from networkx.algorithms.cycles import simple_cycles
import gurobipy as gp
from gurobipy import GRB
import sys
from datetime import datetime


# Step 1: Read the CSV file and create a directed graph
def read_graph_from_csv(file_path):
    #df = pd.read_csv(file_path)
    df=pd.read_csv(file_path, header=None, names=['source', 'destination', 'weight'])
    G = nx.DiGraph()
    for index, row in df.iterrows():
        G.add_edge(row['source'], row['destination'], weight=int(row['weight']))
        #print("source=",row['source'], "dest=",row['destination'], "weight=",row['weight'])
    return G

# Step 2: Find all directed cycles in the graph
def find_cycles(G):
    return list(simple_cycles(G))
    #       next_n_cycles = list(itertools.islice(all_cycles, start, start + number))

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







def dfs_remove_cycle_edges(graph):
    removed_edges = set()
    total_weight = sum(data['weight'] for u, v, data in graph.edges(data=True))
    def dfs(node, stack, visited, rec_stack):
        visited.add(node)
        rec_stack.add(node)
        stack.append(node)
        for neighbor, weight in graph[node].items():
            if neighbor not in visited:
                if dfs(neighbor, stack, visited, rec_stack):
                    return True
            elif neighbor in rec_stack:
                cycle = stack[stack.index(neighbor):] + [neighbor]
                #min_edge = min((u, v) for u, v in zip(cycle, cycle[1:] + [cycle[0]]), key=lambda edge: graph[edge[0]][edge[1]]['weight'])
                #removed_edges.add(min_edge)


                cycle_edges = [(cycle[i], cycle[i + 1]) for i in range(len(cycle) - 1)]
                cycle_weights = [(u, v, graph[u][v]['weight']) for u, v in cycle_edges]
                min_edge = min(cycle_weights, key=lambda x: x[2])
                removed_edges.add(min_edge)

                return True
        stack.pop()
        rec_stack.remove(node)
        return False
    
    visited = set()
    for node in graph.nodes():
        if node not in visited:
            if dfs(node, [], visited, set()):
                break
    #removed_weight = sum(graph[u][v]['weight'] for u, v in removed_edges)
    removed_weight = sum(w for u, v,w in removed_edges)
    return total_weight, removed_weight, removed_edges



def bfs_remove_cycle_edges(G):
    removed_edges = set()
    all_edges_weight = sum(data['weight'] for u, v, data in G.edges(data=True))

    def find_cycle_bfs(start):
        visited = set()
        stack = [(start, [start])]
        while stack:
            (vertex, path) = stack.pop()
            for next_vertex in graph.neighbors(vertex):
                if next_vertex in path:  # Cycle detected
                    cycle = path[path.index(next_vertex):] + [next_vertex]
                    cycle_edges = [(cycle[i], cycle[i + 1]) for i in range(len(cycle) - 1)]
                    # Find the edge with the smallest weight in the cycle
                    #cycle_edges = list(zip(cycle, cycle[1:] + [cycle[0]]))
                    cycle_weights = [(u, v, G[u][v]['weight']) for u, v in cycle_edges]
                    #min_edge = min(cycle_edges, key=lambda edge: G[edge[0]][edge[1]]['weight'])
                    min_edge = min(cycle_weights, key=lambda x: x[2])
                    removed_edges.add(min_edge)
                else:
                    visited.add(next_vertex)
                    stack.append((next_vertex, path + [next_vertex]))

    for node in G.nodes:
        find_cycle_bfs(node)

    #removed_weight = sum(G[u][v]['weight'] for u, v in removed_edges)
    removed_weight = sum(w for u, v, w in removed_edges)

    return all_edges_weight, removed_weight, removed_edges




# Main function to perform all steps
def process_graph(file_path, output_file):
    print("read file")
    G = read_graph_from_csv(file_path)
    total_weight,removed_weight, removed_edges=dfs_remove_cycle_edges(G)
    percentage_removed = (removed_weight / total_weight) * 100

    #print("Removed Edges:", removed_edges)
    print("Total Edge Weight:", total_weight," Total number of Edges=",G.number_of_edges())
    print("Removed Edge Weight:", removed_weight, " Removed number of Edges=", len(removed_edges))
    print("Percentage of Removed Edges Weight:", percentage_removed)

    G_dag = construct_dag(G, removed_edges)
    G_dag_relabelled, mapping = relabel_dag(G_dag)

    write_relabelled_nodes_to_file(mapping, output_file)

# Example usage
file_path = sys.argv[1]
current_time = datetime.now()
time_string = current_time.strftime("%Y%m%d_%H%M%S")

# Create the file name using the formatted time string
output_file = f"relabel_nodes_{time_string}.csv"

process_graph(file_path, output_file)


