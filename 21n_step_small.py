import duckdb

def get_init_node():
    filename="data06_uniq/chembl.init_node.tsv"
    init_nodes=[]
    for line in open(filename):
        arr=line.strip().split("\t")
        init_nodes.append(int(arr[0]))
    print(filename,":", init_nodes[0], len(init_nodes))
    return init_nodes

init_nodes = get_init_node()

con_n = duckdb.connect()
con_n.execute("PRAGMA threads=8")
con_n.execute("PRAGMA memory_limit='16GB'")
con_n.execute("PRAGMA temp_directory='/tmp'")
def check_node(s):
    # Parquetを直接テーブル扱いでクエリできる
    df = con_n.execute("""
    SELECT *
    FROM read_parquet('data08_node/chembl.node.parquet')
    WHERE c1 = {}
    LIMIT 20
    """.format(s)).df()
    print(df)
    #FROM 'data08/bgee.graph.parquet'


#node_start, node_end = 1169459,    1222922730
#node_start, node_end = 1206896772, 1222923884
#1169459 1222922730

con = duckdb.connect()
con.execute("PRAGMA threads=8")
con.execute("PRAGMA memory_limit='16GB'")
con.execute("PRAGMA temp_directory='/tmp'")
import numpy as np
import os
#N=1000000 #=> 3000
N=1000000*3000

os.makedirs("data09",exist_ok=True)
with open("data09/chembl.walk.tsv","w") as ofp:
    for _ in range(N):
        print("==restart==")
        #s=np.random.randint(node_start, node_end+1)
        si=np.random.randint(0, len(init_nodes))
        s=init_nodes[si]

        walk=[]
        walk.append([None,s])
        for k in range(10):
            print("query> ",s)
            df = con.execute("""
            SELECT *
            FROM read_parquet('data08/chembl.graph.parquet')
            WHERE c1 = {}
            LIMIT 100
            """.format(s)).df()
            #FROM 'data08/bgee.graph.parquet'

            if len(df)==0: # terminate
                print("... terminate")
                break
            print(df)
            i=np.random.randint(0, len(df))
            s=int(df.loc[i]["c3"])
            edge=int(df.loc[i]["c2"])

            walk.append([edge,s])

        if len(walk)>2:
            print(walk)
            s="\t".join([",".join(map(str,e)) for e in walk])
            ofp.write(s)
            ofp.write("\n")
            print("===")
        #for e in walk:
        #    check_node(str(e[1]))
