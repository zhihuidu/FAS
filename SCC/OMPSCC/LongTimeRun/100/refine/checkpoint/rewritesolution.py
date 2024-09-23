import sys
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

# Example usage
file_path=sys.argv[1]
process_gurobi_solution(file_path)

