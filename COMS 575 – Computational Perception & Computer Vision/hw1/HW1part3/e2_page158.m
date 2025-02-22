A=[3 -3 4;2 -3 4;0 -1 1];
[V,D] = eig(A)

B=A*A
[V,D2] = eig(B)
D=D*D


D4=D2*D2

A4=B*B


Ainv= A\eye(3)