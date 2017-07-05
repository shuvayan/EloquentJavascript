function minimum(a,b){
	if (a < b)
		return a;
	else 
		return b;
}

function isEven(number){
	if (number == 0)
		return Boolean("True");
	else if (number == 1)
		return Boolean(!"True");
	else if (number == -1)
		number = Number(prompt("Please enter a positive number:"));
	return isEven(number-2);
}