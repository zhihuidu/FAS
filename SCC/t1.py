import sys
# Define the input and output file paths
input_file = sys.argv[1]
output_file = 'new-'+input_file

# Initialize a set to keep track of unique records
unique_records = set()

# Open the input file for reading and the output file for writing
with open(input_file, 'r') as infile, open(output_file, 'w') as outfile:
    for line in infile:
        # Extract the last three columns (ignore the first column)
        columns = line.strip().split(' ', 1)[1]
        
        # Add the columns to the set if not already present
        print(f"old line is {line}")
        print(f"new line is {columns}")
        if columns not in unique_records:
            unique_records.add(columns)
            outfile.write(line)

print(f"Processed file saved as {output_file}")

