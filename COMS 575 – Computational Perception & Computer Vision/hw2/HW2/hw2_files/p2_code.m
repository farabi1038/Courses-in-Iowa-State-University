%insert your code in Part_2/p2_code.cpp
%edit the file extension and web template to match your programing language

I = imread("p1_img.jpg");
BW = im2bw(I,0.5);
BW=~BW;
%imshow(BW);
se1=strel('line',30,0);
se2=strel('line',50,90);
%imshow(BW);
hor= imerode(BW, se1);
hor = imdilate(hor, se1);
%figure; imshow(Apos)


ver= imerode(BW, se2);
res=hor+ver;
imshow(~res);
imwrite(~res, "p2_img1.jpg", "Quality", 100)
elem=(BW-res);
se3=ones(9,1);
elem=imdilate(elem, se3);
elem=imerode(elem, se3);

imshow(~elem);
imwrite(~elem, "p2_img2.jpg", "Quality", 100)


