'''
Usage: feedforward_eval.py <ordered-nodes.csv> <graph.csv>
Example: feedforward_eval.py benchmark.csv connectome_graph.csv
'''
import csv
import sys

edge_flag={}

def process_gurobi_solution(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
    
    # Iterate through each line and process variables
    for line in lines:
        # Split the line to get variable and value
        parts = line.split()
        
        if len(parts) != 2:
            continue  # Skip lines that don't have exactly 2 parts
        
        variable, value = parts[0], parts[1]
        
        # Check if the variable is an edge variable (starts with 'x')
        if variable.startswith('x_'):
            # Extract vertices u and v from the variable name
            # Variable format is x_{u}_{v}, so we split by '_'
            variable_parts = variable.split('_')
            if len(variable_parts) == 3:
                u = variable_parts[1]
                v = variable_parts[2]
                # Output the edge in the desired format: u, v, value
                if abs(float(value) -0.001) <0.1:
                    print(f'{u}, {v}, 0')
                    edge.flag[(u,v)]=0

# Example usage
file_path=sys.argv[1]
process_gurobi_solution(file_path)



node_id_to_index = {}
# reading ordered nodes
# ordered-nodes.csv should be a CSV file with two columns: node_id, order
with open(sys.argv[1], 'r', encoding='utf-8') as f:
    csv_reader = csv.reader(f)
    #next(csv_reader)  # Skip header row
    node_id_to_index = {node: int(order) for (node, order) in csv_reader}

forward_edges_total = 0
all_edges_total = 0
removed_weight = 0

# reading graph
# graph.csv should be a CSV file with three columns: source, target, weight
#with open("removed.csv", 'w') as f2:
with open(sys.argv[2], 'r', encoding='utf-8') as f:
        csv_reader = csv.reader(f)
        next(csv_reader)  # Skip header row
        for row in csv_reader:
            source, target, weight = row
            all_edges_total += int(weight)
            if (u,v) in edge_flag:
                removed_weight+=int(weight)
            else:
                forward_edges_total += int(weight)

print(f'Forward edges: {forward_edges_total}')
print(f'Percent forward: {forward_edges_total/all_edges_total*100}')
print(f'total weight: {all_edges_total}')
print(f'removed weight : {removed_weight}')
print(f'removed percentate : {removed_weight/all_edges_total*100}')
