'''
Usage: feedforward_eval.py <ordered-nodes.csv> <graph.csv>
Example: feedforward_eval.py benchmark.csv connectome_graph.csv
'''
import csv
import sys

node_id_to_index = {}
# reading ordered nodes
# ordered-nodes.csv should be a CSV file with two columns: node_id, order
with open(sys.argv[1], 'r', encoding='utf-8') as f:
    csv_reader = csv.reader(f)
    #next(csv_reader)  # Skip header row
    node_id_to_index = {node: int(order) for (node, order) in csv_reader}

forward_edges_total = 0
all_edges_total = 0

# reading graph
# graph.csv should be a CSV file with three columns: source, target, weight
with open(sys.argv[2], 'r', encoding='utf-8') as f:
    csv_reader = csv.reader(f)
    next(csv_reader)  # Skip header row
    for row in csv_reader:
        source, target, weight = row
        all_edges_total += int(weight)
        if node_id_to_index[source] < node_id_to_index[target]:
            forward_edges_total += int(weight)

print(f'Forward edges: {forward_edges_total}')
print(f'Percent forward: {forward_edges_total/all_edges_total:2.0%}')
