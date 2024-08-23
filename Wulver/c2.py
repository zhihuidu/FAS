import pandas as pd
import networkx as nx
import sys

# Read the CSV file
# Replace 'your_file.csv' with the path to your CSV file
csv_file = sys.argv[1]
data = pd.read_csv(csv_file, header=None, names=['source', 'destination', 'weight'])

# Create a directed graph
G = nx.DiGraph()

# Add edges along with weights
minw=9999
maxw=0
for index, row in data.iterrows():
    G.add_edge(row['source'], row['destination'], weight=row['weight'])
    #print("source=",row['source']," destination=", row['destination'], " weight=", row['weight'])
    if minw >row['weight'] :
        minw=row['weight']
    if maxw <row['weight']:
        maxw=row['weight']
    

# Since triangles are defined for undirected graphs, we need to convert it to an undirected graph
undirected_graph = G.to_undirected()

# Calculate the number of triangles
triangle_count = nx.triangles(undirected_graph)

# Total number of triangles
total_triangles = sum(triangle_count.values()) // 3  # Each triangle is counted 3 times

print(f'Total number of triangles in the graph: {total_triangles}')
print(f'Weight is between {minw} and {maxw} ')

