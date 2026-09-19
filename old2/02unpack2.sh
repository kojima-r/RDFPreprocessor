
for f in `ls data01/**/latest/*.zip`
do
	echo $f
	path=`dirname $f`
	unzip $f -d ${path}
done
