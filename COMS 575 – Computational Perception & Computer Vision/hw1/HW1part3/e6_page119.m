function f = e6_page119(n)
    f = n 

    if n == 0
        f = 1  
    else
        f = f * e6_page119(n-1)
    end 
end
