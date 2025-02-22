
img=imread("p1_search.png");
imgb=im2bw( img,0.5);
imgb=~imgb;

[y,x] = find(imgb) ;
L = max(x)-min(x) ;
B = max(y)-min(y) ;
Icrop = imcrop(imgb,[min(x) min(y) L B]) ;
imshow(~Icrop);