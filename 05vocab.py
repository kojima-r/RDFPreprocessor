import glob
import os
import joblib
from multiprocessing import Pool


def conv_bnode(node_name, filename):
    return filename+"_"+node_name

def run(argv):
    target_dir,out_dir=argv
    node_vocab={}
    edge_vocab={}
    literal_cnt=0
    ofp=open(out_dir+"/literal.tsv","w")
    ofp_node=open(out_dir+"/node.tsv","w")
    ofp_edge=open(out_dir+"/edge.tsv","w")
    ofp_graph=open(out_dir+"/graph.tsv","w")
    for filename in glob.glob(target_dir+"/**/*.tsv",recursive=True):
        #fp=open("data05/biosample/latest/bioschemas.0000.tsv")
        print(">>",filename)
        fp=open(filename)
        for line in fp:
            arr=line.split("\t")
            nt1,nt2,nt3, n1,n2,n3=arr
            if nt3=="Literal":
                if n1 not in node_vocab:
                    n1_i=len(node_vocab)
                    node_vocab[n1]=n1_i
                else:
                    n1_i=node_vocab[n1]
                ofp.write(str(n1_i))
                ofp.write("\n")
                ofp.write(line)
                literal_cnt+=1
            else:
                # BNode
                if nt1=="BNode":
                    n1=conv_bnode(n1,filename)
                if nt3=="BNode":
                    n3=conv_bnode(n3,filename)
                # node set
                if n1 not in node_vocab:
                    n1_i=len(node_vocab)
                    node_vocab[n1]=n1_i
                    ofp_node.write("\t".join([str(n1_i),nt1,n1]))
                    ofp_node.write("\n")
                else:
                    n1_i=node_vocab[n1]
                if n3 not in node_vocab:
                    n3_i=len(node_vocab)
                    node_vocab[n3]=n3_i
                    ofp_node.write("\t".join([str(n3_i),nt3,n3]))
                    ofp_node.write("\n")
                else:
                    n3_i=node_vocab[n3]
                # edge set
                if n2 not in edge_vocab:
                    n2_i=len(edge_vocab)
                    edge_vocab[n2]=n2_i
                    ofp_edge.write("\t".join([str(n2_i),nt2,n2]))
                    ofp_edge.write("\n")
                else:
                    n2_i=edge_vocab[n2]
                ofp_graph.write("\t".join(map(str,[n1_i,n2_i,n3_i])))
                ofp_graph.write("\n")



    print("#node",len(node_vocab))
    print("#edge",len(edge_vocab))
    print("#literal",literal_cnt)

    #with open(target_dir+"/node_vocab.jbl", mode="wb") as ofp:
    #    joblib.dump(node_vocab, ofp, compress=3)
    #with open(target_dir++"/edge_vocab.jbl", mode="wb") as ofp:
    #    joblib.dump(edge_vocab, ofp, compress=3)



if __name__ == "__main__":
    target="data04/*"
    data=[]
    for path in glob.glob(target):
        print(path)
        name=os.path.basename(path)
        out_path="data05/"+name
        os.makedirs(out_path,exist_ok=True)
        data.append((path, out_path))
    p = Pool(1)
    p.map(run, data)
