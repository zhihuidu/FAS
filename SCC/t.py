import re
import sys

def remove_non_matching_lines(filename,newname):
    with open(filename, 'r') as file1:
        lines = file1.readlines()
        with open(newname, 'a') as file2:
            for line in lines:
                if re.match(r"^removed \d+", line.strip()):  # Check if the line starts with 'removed' followed by digits
                    file2.write(line)

# Example usage
filename = sys.argv[1]
newname = 'new'+filename
remove_non_matching_lines(filename,newname)

