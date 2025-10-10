from neo4j import GraphDatabase

URI = "neo4j://localhost:7687"   # 単ノードなら bolt:// でもOK
AUTH = ("neo4j", "neo4jneo4j")

EXPORT_DIR_FILENAMES = {
    "undirected": "degree_distribution_undirected.csv",
    "out": "degree_distribution_out.csv",
    "in": "degree_distribution_in.csv",
}

# ----- Cypher templates -----

# 1) 投影（存在しなければ作る）: 無向（総次数）
CREATE_GRAPH_UNDIRECTED = """
CALL gds.graph.exists('g_entity_undirected') YIELD exists
WITH exists
CALL apoc.do.when(
  exists,
  'RETURN "exists" AS status',
  '
   CALL gds.graph.project(
     "g_entity_undirected",
     { Entity: { label: "Entity" } },
     { ANY: { type: "*", orientation: "UNDIRECTED" } }
   ) YIELD graphName
   RETURN graphName AS status
  ',
  {}
) YIELD value
RETURN value.status AS status;
"""

# 2) 投影（存在しなければ作る）: 有向（入/出次数）
CREATE_GRAPH_DIRECTED = """
CALL gds.graph.exists('g_entity_directed') YIELD exists
WITH exists
CALL apoc.do.when(
  exists,
  'RETURN "exists" AS status',
  '
   CALL gds.graph.project(
     "g_entity_directed",
     { Entity: { label: "Entity" } },
     { ANY: { type: "*", orientation: "NATURAL" } }
   ) YIELD graphName
   RETURN graphName AS status
  ',
  {}
) YIELD value
RETURN value.status AS status;
"""

# 3) エクスポート（総次数: 無向）
EXPORT_UNDIRECTED = """
CALL apoc.export.csv.query(
  "
  CALL gds.degree.stream('g_entity_undirected')
  YIELD score
  WITH toInteger(score) AS k
  RETURN k AS degree, count(*) AS freq
  ORDER BY degree
  ",
  $filename,
  {batchSize:200000, quotes:false}
) YIELD file, rows, time
RETURN file, rows, time;
"""

# 4) エクスポート（出次数）
EXPORT_OUT = """
CALL apoc.export.csv.query(
  "
  CALL gds.degree.stream('g_entity_directed', {orientation:'NATURAL'})
  YIELD score
  WITH toInteger(score) AS kout
  RETURN kout AS out_degree, count(*) AS freq
  ORDER BY out_degree
  ",
  $filename,
  {batchSize:200000, quotes:false}
) YIELD file, rows, time
RETURN file, rows, time;
"""

# 5) エクスポート（入次数）
EXPORT_IN = """
CALL apoc.export.csv.query(
  "
  CALL gds.degree.stream('g_entity_directed', {orientation:'REVERSE'})
  YIELD score
  WITH toInteger(score) AS kin
  RETURN kin AS in_degree, count(*) AS freq
  ORDER BY in_degree
  ",
  $filename,
  {batchSize:200000, quotes:false}
) YIELD file, rows, time
RETURN file, rows, time;
"""

def main():
    driver = GraphDatabase.driver(URI, auth=AUTH)
    with driver.session() as session:
        # 投影を作成（初回だけ実行され、以降はスキップ）
        print(session.run(CREATE_GRAPH_UNDIRECTED).single()["status"])
        print(session.run(CREATE_GRAPH_DIRECTED).single()["status"])

        # 総次数（無向）
        rec = session.run(EXPORT_UNDIRECTED, filename=EXPORT_DIR_FILENAMES["undirected"]).single()
        print("[undirected]", rec["file"], "rows:", rec["rows"], "time(ms):", rec["time"])

        # 出次数
        rec = session.run(EXPORT_OUT, filename=EXPORT_DIR_FILENAMES["out"]).single()
        print("[out]", rec["file"], "rows:", rec["rows"], "time(ms):", rec["time"])

        # 入次数
        rec = session.run(EXPORT_IN, filename=EXPORT_DIR_FILENAMES["in"]).single()
        print("[in]", rec["file"], "rows:", rec["rows"], "time(ms):", rec["time"])

    driver.close()

if __name__ == "__main__":
    main()
