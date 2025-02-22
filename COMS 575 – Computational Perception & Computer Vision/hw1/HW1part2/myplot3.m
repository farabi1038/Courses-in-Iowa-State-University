x=linspace(-3*pi,3*pi,100)
y=-sin(x)./x;
area(x,y)
xlabel('x'),ylabel('sinx(x)./x')
hold on
x1=x(46:55);y1=y(46:55)
area(x1,y1,'facecolor','y')