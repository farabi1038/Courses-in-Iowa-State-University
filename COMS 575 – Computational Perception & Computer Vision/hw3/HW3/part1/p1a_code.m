%% Your code goes in this file. 


%
clear all
close all

rgb = imread('/Users/ibnefarabishihab/Desktop/Course materials /COMS 575/HW3/part1/p1_image2.png');

%bw = im2bw(img);
% % find both black and white regions
% stats = [regionprops(bw); regionprops(not(bw))]
% % show the image and draw the detected rectangles on it
% imshow(bw); 
% hold on
% for i = [63,66]
%     rectangle('Position', stats(i).BoundingBox, ...
%     'Linewidth', 3, 'EdgeColor', 'r', 'LineStyle', '--');
% end

[centers,radii,metricBright] = imfindcircles(rgb,[15 100],'ObjectPolarity','bright');
imshow(rgb)
h = viscircles(centers,radii);


%imshow(rgb)
% d = drawline;
% pos = d.Position;
% diffPos = diff(pos);
% diameter = hypot(diffPos(1),diffPos(2))


