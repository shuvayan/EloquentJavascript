function myRange(start,end,step){
  var rangeArray = [];
  for (var i = start; step > 1 || step === undefined ? i <= end : i >= end; step ? i = i + step : i++){
    rangeArray.push(i);
    };
  return rangeArray;
};

//var myRange = myRange(1,10)

function mySum(sumArray){
	return sumArray.reduce(function(x,y){
    return x+y;
  });
};

//console.log(mySum(myRange(1,10,2)))
