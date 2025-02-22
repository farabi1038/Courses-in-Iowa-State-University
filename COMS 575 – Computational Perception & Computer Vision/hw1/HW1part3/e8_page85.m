A=rand(10)
A=A*100
A=fix(A)
A( A < 10 ) = 0
A( A > 90 ) = inf
idx = (A>=30) & (A<=50);
b= A(idx)