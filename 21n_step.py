import duckdb

con_n = duckdb.connect()
con_n.execute("PRAGMA threads=8")
con_n.execute("PRAGMA memory_limit='16GB'")
con_n.execute("PRAGMA temp_directory='/tmp'")
def check_node(s):
    # Parquetを直接テーブル扱いでクエリできる
    df = con_n.execute("""
    SELECT *
    FROM read_parquet('data08_node/*.node.parquet')
    WHERE c1 = {}
    LIMIT 20
    """.format(s)).df()
    print(df)
    #FROM 'data08/bgee.graph.parquet'

con = duckdb.connect()
con.execute("PRAGMA threads=8")
con.execute("PRAGMA memory_limit='16GB'")
con.execute("PRAGMA temp_directory='/tmp'")
import numpy as np
for _ in range(3):
    print("==restart==")
    s=np.random.randint(521670, 3196728727+1)

    walk=[]
    walk.append([None,s])
    for k in range(10):
        print(s)
        df = con.execute("""
        SELECT *
        FROM read_parquet('data08/*.graph.parquet')
        WHERE c1 = {}
        LIMIT 100
        """.format(s)).df()
        print(df)
        #FROM 'data08/bgee.graph.parquet'

        if len(df)==0: # terminate
            break
        i=np.random.randint(0, len(df))
        s=int(df.loc[i]["c3"])
        edge=int(df.loc[i]["c2"])

        walk.append([edge,s])

    print("===")
    print(walk)
    for e in walk:
        check_node(str(e[1]))
