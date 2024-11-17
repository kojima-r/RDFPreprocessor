import glob
import os
import joblib

node_vocab=joblib.load("node_vocab.jbl")
edge_vocab=joblib.load("edge_vocab.jbl")
ofp=open("graph_all.tsv","w")
target="data05/**/*.tsv"
for filename in glob.glob(target,recursive=True):
    #fp=open("data05/biosample/latest/bioschemas.0000.tsv")
    print(">>",filename)
    fp=open(filename)
    #TODO: BNode未処理
    for line in fp:
        arr=line.split("\t")
        nt1,nt2,nt3, n1,n2,n3=arr
        if nt3=="Literal":
            pass
        else:
            if n1 not in node_vocab:
                pass
            if n3 not in node_vocab:
                pass
            if n2 not in edge_vocab:
                pass

            i1=node_vocab[n1]
            i3=node_vocab[n3]
            i2 =edge_vocab[n2]
            
            arr=[nt1,nt2,nt3, str(i1), str(i2),str(i3)]
            ofp.write("\t".join(arr))
            ofp.write("\n")
        
