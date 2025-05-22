# RDFPreprocessor
生命科学関連RDF ([RDF portal][https://rdfportal.org/]) からのデータセット構築をするプロジェクトで、以下ファイルから構成される

各並列化数などはスクリプトに直書きされているので必要に応じて書き換える必要がある（TODO:設定ファイル化）

## 環境
```
pip install beautifulsoup4
pip install rdflib
```

## Script: 00
- RDF portal からダウンロード予定のRDF一覧を作成する

```
python 00make_list.py
```

## Script: 01 
- RDF portal からのRDFをダウンロードスクリプト
（途中で落ちた場合は基本的には途中から再開できる）

```
python 01download.py 
```

## Script: 02
-02unpackは圧縮ファイルの展開
### ファイルの分割
安全のため分割は手動で行うようになっている（TODO:自動化）．

参考：10GB以上のファイルの検索コマンド
```
find ./ -size +10G
```

ttlファイルの分割スクリプトに分割するファイルの一覧を直書きする必要がある．
分割後はfix_ttl2という拡張子を付ける

```
python 02split_ttl.py
```

分割元のファイルの削除は自動では安全のため手動で行うようになっている
TODO:元ファイルのファイル名を<元ファイル>.ttl_originalのように拡張子を自動的に変更する（拡張子を変えないと以降で自動認識されるため）

## Script: 03
- ファイルフォーマットをntripleへと変換
- xml/ttl/rdfなどのデータをntriplへと変換する（破損等がないかのチェックを含む）
- 破損等で途中で止まった場合は再度実行すれば途中から再開できる
```
python 03conv_<xml/ttl/rdf/ttl2>.py
```
ただし，ttl2は分割後のfix_ttl2ファイルを対象とした変換

## Script: 04
- ファイルフォーマットをntripleからtsvへと変換
```
python 04to_tsv.py
```

## Script: 05-07
- 取得したグラフにおいて基本的な性質のチェックスクリプト
- 取得したグラフにおいてグラフ上でのnode2vecスクリプト

