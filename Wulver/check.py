import networkx as nx

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
    # Create a directed graph
    G = nx.DiGraph()

    # Add edges to the graph (example graph)
    G.add_edges_from([(1, 2), (2, 3), (3, 1), (4, 1), (4, 2), (4, 3)])

    # Specify X and B values
    X = 2
    B = 3

    # Get counts
    more_than_X, more_than_B = count_vertices_by_degree(G, X, B)
    print(f"Vertices with degree more than {X}: {more_than_X}")
    print(f"Vertices with degree more than {B}: {more_than_B}")

