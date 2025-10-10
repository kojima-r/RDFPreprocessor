n=3196732692
out_filename="node_id.tsv"
with open(out_filename,"w") as ofp:
    ofp.write("id:ID(LONG)\n")
    for i in range(n+1):
        ofp.write(str(i))
        ofp.write("\n")

