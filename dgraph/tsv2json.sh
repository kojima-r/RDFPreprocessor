mkdir -p ./data_json/
for f in ./data/*.tsv; do
  base=$(basename "$f" .tsv.gz)
  python tsv2json.py $f \
      --type-name Person \
      --id-col id \
      --int-cols age \
  | gzip > ./data_json/${base}.json.gz
done

