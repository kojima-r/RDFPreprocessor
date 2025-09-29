import glob
import os
import joblib

def conv_bnode(node_name, filename):
    return filename+"_"+node_name
node_vocab={}
edge_vocab={}
literal_cnt=0

if __name__ == "__main__":
    data=[]
    target="data04/**/*.tsv"
    ofp=open("literal.tsv","w")
    for filename in glob.glob(target,recursive=True):
        #fp=open("data05/biosample/latest/bioschemas.0000.tsv")
        print(">>",filename)
        fp=open(filename)
        #TODO: BNode未処理
        for line in fp:
            arr=line.split("\t")
            nt1,nt2,nt3, n1,n2,n3=arr

            # BNode
            if nt1=="BNode":
                n1=conv_bnode(n1,filename)
            if nt3=="BNode":
                n3=conv_bnode(n3,filename)
            if nt3=="Literal":
                ofp.write(line)
                literal_cnt+=1
            else:
                if n1 not in node_vocab:
                    node_vocab[n1]=len(node_vocab)
                if n3 not in node_vocab:
                    node_vocab[n3]=len(node_vocab)

                if n2 not in edge_vocab:
                    edge_vocab[n2]=len(edge_vocab)


    print("#node",len(node_vocab))
    print("#edge",len(edge_vocab))
    print("#literal",literal_cnt)


with open("node_vocab.jbl", mode="wb") as ofp:
    joblib.dump(node_vocab, ofp, compress=3)
with open("edge_vocab.jbl", mode="wb") as ofp:
    joblib.dump(edge_vocab, ofp, compress=3)

