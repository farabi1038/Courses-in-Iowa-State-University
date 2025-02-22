%insert your code in Part_1/p1_matlab_code.m
%{
one way
img=imread("p1_search.png");
imgb=im2bw( img,0.5);
imgb=~imgb;
[L,num]=bwlabel(imgb,8);
stats=regionprops(imgb,'Centroid','Area','BoundingBox');
center=cat(1,stats.Centroid);
box=stats.BoundingBox;
len=length(stats);
lastBox=stats(len).BoundingBox;
I2 = imcrop(img,box);
figure;imshow(I2);
figure;imshow(img);
imwrite(I2, "p1_img.jpg", "Quality", 100)
%}

%another way
img=imread("p1_search.png");
imgb=im2bw( img,0.5);
imgb=~imgb;

[y,x] = find(imgb) ;
L = max(x)-min(x) ;
B = max(y)-min(y) ;
Icrop = imcrop(imgb,[min(x) min(y) L B]) ;
imshow(~Icrop);
imwrite(~Icrop, "p1_img.jpg", "Quality", 100)



