echo "addition of numbers until n"
read -p "enter number n :" n
echo "The addition is $((n*(n+1)/2))"

echo "factorial of numbers"
val=1
n2=n
while [ $n -gt 0 ];do
	val=$((val*n))
	n=$((n-1))
done

echo "the factorial number of $n2 : $val"

