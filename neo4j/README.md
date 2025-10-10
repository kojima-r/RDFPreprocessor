 sudo apt install neo4j
 wget -O - https://debian.neo4j.com/neotechnology.gpg.key | sudo apt-key add -
 echo 'deb https://debian.neo4j.com stable latest' | sudo tee -a /etc/apt/sources.list.d/neo4j.list
 deb https://debian.neo4j.com stable latest
 sudo apt-get update
 apt list -a neo4j
 sudo apt-get install neo4j


sudo cp -rp /var/lib/neo4j/ /data4/
sudo rm -r /var/lib/neo4j
sudo ln -s /data4/neo4j/ /var/lib/neo4j



service neo4j start

http://127.0.0.1:7474/browser/

neo4jneo4j

chmod 777 .

sudo -u neo4j sh run.sh

```
$ cypher-shell -u neo4j -p 'neo4jneo4j' -d neo4j "

CALL gds.graph.drop('myGraph', false);

CALL gds.graph.project.cypher(
  'myGraph',
  'MATCH (n) RETURN id(n) AS id',
  'MATCH (s)-[r]->(t)
   RETURN id(s) AS source, id(t) AS target, type(r) AS type'
);

// ランダム開始ノードから 10 ノード（= 9 ステップ）
MATCH (n) WITH id(n) AS startId ORDER BY rand() LIMIT 1
CALL gds.randomWalk.stream('myGraph', {
  sourceNodes: [startId],
  walkLength: 3,     // 10ノード = 9ステップ
  walksPerNode: 1
})
YIELD nodeIds
RETURN [nid IN nodeIds | gds.util.asNode(nid)] AS nodes;
"
+--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| graphName | database | databaseLocation | memoryUsage | sizeInBytes | nodeCount | relationshipCount | configuration | density | creationTime | modificationTime | schema | schemaWithOrientation |
+--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
+--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+

0 rows
ready to start consuming query after 5 ms, results consumed after another 2 ms
+-------------------------------------------------------------------------------------------+
| nodeQuery | relationshipQuery | graphName | nodeCount | relationshipCount | projectMillis |
+-------------------------------------------------------------------------------------------+
52U00: procedure exception - custom procedure execution error cause. Execution of the procedure gds.graph.project.cypher() failed due to java.lang.OutOfMemoryError: Java heap space.
  52N37: procedure exception - procedure execution error. Execution of the procedure gds.graph.project.cypher() failed.

```
