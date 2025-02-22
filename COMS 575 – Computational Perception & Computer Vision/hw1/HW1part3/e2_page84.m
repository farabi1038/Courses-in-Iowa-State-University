A=[2 6;3 9 ];
B=[1 2;3 4 ];
C=[-5 5;5 3 ];
if (A+B)==(B+A)
    disp("Commutative :yes")
end 

if (A+B)+C==(A+B)+C
    disp("associative:yes")
end 

if 5*(A+B)==(5*A+5*B)
    disp("multiplication distributive :yes")
end 

if (A*B)==(A*C)
    disp("matrix are different than scalar:no")
else
    disp("matrix are different than scalar:yes")
end

