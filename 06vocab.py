import glob
import os
import joblib

def run(argv):
    target_dir,out_dir=argv
    node_vocab={}
    edge_vocab={}
    literal_cnt=0
    ofp=open(out_dir+"/literal.tsv","w")
    for filename in glob.glob(target_dir+"/**/*.tsv",recursive=True):
        #fp=open("data05/biosample/latest/bioschemas.0000.tsv")
        print(">>",filename)
        fp=open(filename)
        #TODO: BNode未処理
        for line in fp:
            arr=line.split("\t")
            nt1,nt2,nt3, n1,n2,n3=arr
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

    with open(target_dir+"/node_vocab.jbl", mode="wb") as ofp:
        joblib.dump(node_vocab, ofp, compress=3)
    with open(target_dir++"/edge_vocab.jbl", mode="wb") as ofp:
        joblib.dump(edge_vocab, ofp, compress=3)



if __name__ == "__main__":
    target="data05/*"
    for path in glob.glob(target):
        #for filename in glob.glob(target,recursive=True):
        print(path)
        name=os.path.basename(path)
        out_path="data06/"+name
        #os.makedirs(path1,exist_ok=True)
        data.append((path, out_path))
    p = Pool(8)
    p.map(run, data)
