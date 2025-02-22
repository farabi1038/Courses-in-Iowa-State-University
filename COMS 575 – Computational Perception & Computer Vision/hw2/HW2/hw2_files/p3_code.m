%insert your code in Part_3/p3_code.cpp
%edit the file extension and web template to match your programing language

I = imread("p2_img1.jpg");
BW = im2bw(I,0.5);
BW=~BW;
imshow(BW);
se3=ones(62,58);
se4=ones(65,61);
elem=imdilate(BW, se3);
%elem=imerode(elem, se3);
imshow(elem);
