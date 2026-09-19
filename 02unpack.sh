for f in `ls data01/**/latest/*.gz`
do
	echo $f
	gunzip $f
done
for f in `ls data01/**/**/latest/*.gz`
do
	echo $f
	gunzip $f
done
