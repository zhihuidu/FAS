import networkx as nx

# Step 1: Create a directed graph
G = nx.DiGraph()

# Step 2: Add nodes and edges
G.add_edges_from([('A', 'B'), ('A', 'C'), ('B', 'C'), ('B', 'D'), ('C', 'D'), ('D', 'E'), ('E', 'A'), ('C', 'E'), ('E', 'B')])

# Step 3: Calculate the sum of in-degree and out-degree for each node
degree_sum = {node: G.in_degree(node) + G.out_degree(node) for node in G.nodes()}
print(f"degree sum is {degree_sum}")

# Step 4: Sort nodes by the degree sum in descending order
sorted_vertices = sorted(degree_sum.items(), key=lambda x: x[1], reverse=True)

print(f"sorted vertices is {sorted_vertices}")
# Step 5: Select the top K heavy vertices (e.g., top 3)
K = 3
top_k_heavy_vertices = [node for node, degree in sorted_vertices[:K]]

# Step 6: Print the top K heavy vertices
print(f"Top {K} heavy vertices based on sum of in-degree and out-degree:")
print(top_k_heavy_vertices)
print(f"degree sum of the kth node is {sorted_vertices[K][1]}")

