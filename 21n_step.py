import duckdb

con = duckdb.connect()
con.execute("PRAGMA threads=8")
con.execute("PRAGMA memory_limit='16GB'")
con.execute("PRAGMA temp_directory='/tmp'")
import numpy as np
for _ in range(3):
    print("==restart==")
    s=np.random.randint(521670, 3196728727+1)

    walk=[]
    walk.append(s)
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
