I = imread("p1_img.jpg");
BW = im2bw(I,0.5);
BW=~BW;
se = imread("Symbol_Cutouts/X.png");
se = im2bw(se);
se=~se;
res=imerode(BW,se);
res=imdilate(res,se);
imshow(res);