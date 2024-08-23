import networkx as nx

# Create or load your directed graph
G = nx.DiGraph()

# Add edges to the graph (example)
G.add_edges_from([(1, 2), (2, 3), (3, 1), (4, 5),(5,7),(5,1),(7,6),(6,1),(3,4)])

# Compute strongly connected components
scc = list(nx.strongly_connected_components(G))

# Print the strongly connected components
print("Strongly Connected Components:", scc)

