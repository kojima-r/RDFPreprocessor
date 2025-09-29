filename="data05/bgee/shared_node.tsv"
data=[]
for line in open(filename):
    arr=line.strip().split("\t")
    data.append(arr)


with open(filename,"w") as ofp:
    for arr in data:
        if arr[1]!="0":
            ofp.write("\t".join(arr))
            ofp.write("\n")

