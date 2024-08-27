import re
import sys

def remove_non_matching_lines(filename,newname):
    with open(filename, 'r') as file1:
        lines = file1.readlines()
        with open(newname, 'w') as file2:
            for line in lines:
                    newline=f"{line},0\n"
                    file2.write(newline)

# Example usage
filename = sys.argv[1]
newname = 'new'+filename
remove_non_matching_lines(filename,newname)

