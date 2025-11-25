 
import os 
import glob
import numpy as np
def run_out(name):
    filename="data06/{}.graph.tsv".format(name)
    out_list={}
    for line in open(filename):
        arr=line.strip().split("\t")
        key=(int(arr[0]),int(arr[1]))
        if key not in out_list:
            out_list[key]=[]
        out_list[key].append((int(arr[2])))
    return out_list

import shelve

def run_out2(name):
    filename = f"data06/{name}.graph.tsv"
    db_path = f"cache/cache_{name}.db"
    with shelve.open(db_path, flag='n', writeback=False) as db:
        with open(filename) as f:
            for line in f:
                a, b, c = map(int, line.split("\t"))
                key = f"{a}_{b}"

                 # 既存リストを取り出して更新する
                lst = db.get(key)
                if lst is None:
                    db[key] = [c]
                else:
                    lst.append(c)
                    db[key] = lst  # ← writeback=False の場合は必ず再代入が必要

        # メモリに乗らずディスク上で参照可能

        out_filename="data07/{}.graph.tsv".format(name)
        with open(out_filename, "w") as ofp:
            # db.items() でもいいですが、値が大きい場合を考えて key から手動で辿っています
            #for k,v in db.items():
            for k in db.keys():
                v = db[k]  # この時点でそのキーのリストだけがロードされる
                s="\t".join(map(str, v))
                k_s=k.split("_")
                ofp.write(str(k_s[0]))
                ofp.write("\t")
                ofp.write(str(k_s[1]))
                ofp.write("\t")
                ofp.write(s)
                ofp.write("\n")
    return


os.makedirs("data07",exist_ok=True)

target="data06/*.graph.tsv"
for filename in glob.glob(target,recursive=True):
    bname=os.path.basename(filename)
    name1,_ = os.path.splitext(bname)
    name,_ = os.path.splitext(name1)
    print("... running:",name)
    run_out2(name)

