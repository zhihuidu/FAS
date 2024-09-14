for i in `ls *.err`  
do
	echo "errors in $i are as follows"
	cat "$i"
done
for i in `ls *.out`  
do
	echo "output in $i are as follows"
	cat "$i"
done
