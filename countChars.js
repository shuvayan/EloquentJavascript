// 1. Simple if-else.
function countBs (text,char){
	string_var = String(text),char_var = String(char),count_char = 0;
	for (i = 0;i <= string_var.length ; i++){
		if (string_var.charAt(i) == char_var){
			count_char += 1;
        }
		else {
			count_char += 0;
        }
    }
	return count_char;
}

//.Using Ternary operator:
function countBs (text,char){
	string_var = String(text),char_var = String(char),count_char = 0;
	for (i = 0;i <= string_var.length ; i++){
		count_char = ((string_var.charAt(i) == char_var) ? count_char += 1 : count_char += 0);
        }
    return count_char;
}
