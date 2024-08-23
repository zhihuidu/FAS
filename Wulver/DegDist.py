import pandas as pd
import networkx as nx
import sys

def create_graph_from_csv(file_path):
    # Read the CSV file
    df = pd.read_csv(file_path, header=None, names=['source', 'destination', 'weight'])
    #df = pd.read_csv(file_path);

    # Create a directed graph
    G = nx.DiGraph()

    # Add edges to the graph
    for index, row in df.iterrows():
        G.add_edge(row['source'], row['destination'], weight=row['weight'])
    print("Build the graph")
    return G

def count_vertices_by_degree(graph, x, b):
    # Calculate in-degrees and out-degrees
    in_degrees = dict(graph.in_degree())
    out_degrees = dict(graph.out_degree())

    # Total degrees
    total_degrees = {node: in_degrees[node] + out_degrees[node] for node in graph.nodes}

    # Count vertices with degree > x and degree > b
    count_more_than_x = sum(1 for degree in total_degrees.values() if degree > x)
    count_more_than_b = sum(1 for degree in total_degrees.values() if degree > b)

    return count_more_than_x, count_more_than_b

# Example usage
if __name__ == "__main__":
    # Path to the CSV file
    file_path = sys.argv[1];

    # Create the graph
    G = create_graph_from_csv(file_path)

    # Specify X and B values
    X = 400
    Y = 800

    # Get counts
    print("Calculate the values")
    more_than_X, more_than_Y = count_vertices_by_degree(G, X, Y)
    print(f"Vertices with degree more than {X}: {more_than_X}")
    print(f"Vertices with degree more than {Y}: {more_than_Y}")

