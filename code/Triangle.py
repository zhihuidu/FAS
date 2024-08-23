import pandas as pd
import networkx as nx
import sys
# Read the CSV file into a DataFrame
# Replace 'your_file.csv' with your actual CSV file path
csv_file = sys.argv[1]
df = pd.read_csv(csv_file, header=None, names=['source', 'destination', 'weight'])

# Create a directed graph from the DataFrame
G = nx.DiGraph()

# Add edges to the graph from the DataFrame
for index, row in df.iterrows():
    G.add_edge(row['source'], row['destination'], weight=row['weight'])

# Function to count directed triangles
def count_directed_triangles(graph):
    directed_triangles = 0
    # Iterate through all edges in the graph
    for u, v in graph.edges():
        # Check for a common neighbor that completes a triangle
        for w in graph.neighbors(v):
            if graph.has_edge(w, u):
                directed_triangles += 1
    return directed_triangles // 3  # Each triangle is counted 3 times

# Calculate the total number of directed triangles
total_triangles = count_directed_triangles(G)

print(f'Total number of directed triangles: {total_triangles}')

