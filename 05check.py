"""
fp=open("data05/biosample/latest/bioschemas.0000.tsv")
for line in fp:
    arr=line.split("\t")
    if len(arr)!=6:
        print(arr)
        break
quit()
"""
node_vocab={}
edge_vocab={}
literal_edge=[]
#fp=open("data05/biosample/latest/bioschemas.0000.tsv")
#fp=open("data05/eco/latest/eco.tsv")
#fp=open("data05/owl/latest/owl.tsv")
fp=open("test.tsv")
#TODO: BNode未処理
cnt=0
for line in fp:
    arr=line.split("\t")
    if len(arr)!=6:
        print(cnt,arr)
        #cnt+=1
    cnt+=1
    nt1,nt2,nt3, n1,n2,n3=arr
    if nt3=="Literal":
        literal_edge.append(line)
    else:
        if n1 not in node_vocab:
            node_vocab[n1]=len(node_vocab)
        if n3 not in node_vocab:
            node_vocab[n3]=len(node_vocab)

        if n2 not in edge_vocab:
            edge_vocab[n2]=len(edge_vocab)


print("#node",len(node_vocab))
print("#edge",len(edge_vocab))
print("#literal",len(literal_edge))

#with open("vocab.jbl", mode="wb") as ofp:
#    joblib.dump(node_vocab, ofp, compress=3)


