import sys
import csv
with open('test.txt', 'r') as file:
              reader = csv.reader(file)
              for row in reader:
                 print(f"row={row}")
                 x1,x2  = row # Strip any leading/trailing whitespace
                 print(f"x1={x1},x2={x2}")
                 x11,x12=x1.split(maxsplit=1)
                 print(f"x11={x11},x12={x12}")
                 x12=int(x12)
                 x2=int(x2)

