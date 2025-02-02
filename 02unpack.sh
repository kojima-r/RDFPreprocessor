for f in `ls data01/**/latest/*.gz`
do
	echo $f
	gunzip $f
done
