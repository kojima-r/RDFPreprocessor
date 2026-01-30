mkdir -p data06_uniq
for f in `ls data06_sort/*.tsv`
do
	echo $f
	name=`basename $f`
	uniq $f  data06_uniq/$name
done

