mkdir -p data06_sort
for f in `ls data06/*.tsv`
do
	echo $f
	name=`basename $f`
	sort -n $f > data06_sort/$name
done
