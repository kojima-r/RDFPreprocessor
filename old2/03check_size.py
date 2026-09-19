import glob
import os
mode="rdf"
#mode="ttl"
#mode="xml"
output_path="data03"
if __name__ == "__main__":
    data=[]
    if mode=="xml":
        target="data01/**/*.xml"
    elif mode=="ttl":
        target="data01/**/*.ttl"
    elif mode=="rdf":
        target="data01/**/*.rdf"
    for filename in glob.glob(target,recursive=True):
        path=os.path.dirname(filename)
        name=os.path.basename(filename)
        path1=output_path+path[6:]
        os.makedirs(path1,exist_ok=True)
        filename = path+"/"+name
        name_, _=os.path.splitext(name)
        out_filename = path1+"/"+name_+".nt"
        size=os.path.getsize(filename)
        if size>10000000000:
            print(size,filename, out_filename)
            #data.append((filename, out_filename))
    #p = Pool(8) # プロセス数を4に設定
    #p.map(run, data)

