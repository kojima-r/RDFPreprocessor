import numpy as np
import pymetis
import os

#https://github.com/inducer/pymetis
filename="data06/bgee.graph.tsv"
out_filename="stat06/bgee.metis.tsv"
os.makedirs("stat06",exist_ok=True)
max_n1=0
for line in open(filename):
    arr=line.strip().split("\t")
    n1=int(arr[0])
    if max_n1<n1:
        max_n1=n1
    

adjacency_list=[[] for _ in range(max_n1+1)]
for line in open(filename):
    arr=line.strip().split("\t")
    n1=int(arr[0])
    n2=int(arr[2])
    adjacency_list[n1].append(n2)

print("... start metis")
n_cuts, membership = pymetis.part_graph(2, adjacency=adjacency_list)
# n_cuts = 3
# membership = [1, 1, 1, 0, 1, 0, 0]
print(membership)
with open(out_filename,"w") as fp:
    for k in membership:
        fp.write(str(k))
        fp.write("\n")

