import glob
import os
from multiprocessing import Pool



#idsra:SRR7345670 a dra:Run ;
# dct:identifier "SRR7345670" ;
# rdfs:seeAlso idsra:SRA721981 ;
def fix_ttl(filename, out_filename):
    pred=None
    with open(out_filename,"w") as ofp:
        for line in open(filename):
            if pred is not None and len(line.strip())>0:
                if pred[0]==" " and line[0]!=" ":
                    if pred[-1]==";":
                        ofp.write(pred[:-1]+".\n")
                    else:
                        ofp.write(pred)
                        ofp.write("\n")
                else:
                    ofp.write(pred)
                    ofp.write("\n")
                pred=line.rstrip()
            elif pred is None:
                pred=line.rstrip()
            else:
                # skip
                ofp.write("\n")

        if pred is not None:
            ofp.write(pred)
            ofp.write("\n")

def run(argv):
    filename, out_filename = argv
    print(">>",filename)
    fix_ttl(filename, out_filename)


if __name__ == "__main__":
    data=[]
    for filename in glob.glob("data01/**/*.ttl",recursive=True):
        path=os.path.dirname(filename)
        name=os.path.basename(filename)
        filename = path+"/"+name
        out_filename = path+"/"+name+".fix_ttl"
        data.append((filename, out_filename))
    p = Pool(8) # プロセス数を4に設定
    p.map(run, data)


