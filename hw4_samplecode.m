% HW4 Sample codes 
% Lena Again, Image decimation and interpolation
%
%                                               Edited by Meng-Lin Li, 04/21/2016

clear all; close all;% clc;
xc = double(imread('RedTileHouse.jpg')); % convert to "double" data type, in MATLAB, all the processing only available for "double"
xc_size = size(xc);

figure
image(xc);	% you may try MATLAB function image() or imshow()
colormap(gray(256)) % or colormap(gray)
axis image

D = 2; % downsample ratio

%(a) 
% downsample xc according to the downsample ratio
x = uint8(xc(1:D:end, 1:D:end)); 

% interpolate x to the original size, see xc_size
% use the MATLAB function, imresize()
xs = uint8(imresize(x,xc_size,'bicubic')); % imresize(), IMRESIZE(A, [NUMROWS NUMCOLS], METHOD)

%
% Display images xc, xs, and x and comment on their appearance.
%
figure
imshow(x,[0 255])
axis image
figure
imshow(xs,[0 255])
title(gca,'Upsampling and interpolation without anti-aliasing')
axis image
%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%% (b) %%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%
% Blur the image using a moving average filter
% determine a proper M of your MxM MA filter
M = 4; % the value of M
ma = ones(M,M)/(M*M); % filter, impulse response
%%%%%%%%% section for MAF fourier analysis %%%%%%%%%%%
%ma_t = zeros(size(xc));
%ma_t(1:M,1:M)=ma;
%ma_fft = fftshift(fft2(ma_t));
%figure
%imshow(abs(ma_fft))
%figure
%plot(abs(ma_fft(:,270)));
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% The 2D filtering can be done with MATLAB function conv2() or filter2()
% !!! remember to specify the right 'shape' when using conv2() or filter2()
yc = conv2(xc,ma,'same');

% downsample yc
y = yc(1:D:end, 1:D:end);

% rescale y to the original size as yc
% use the MATLAB function, imresize()
ys = imresize(y,xc_size,'bicubic');
figure
imshow(yc,[0 255])
figure
imshow(y,[0 255])
figure
imshow(ys,[0 255])
title(gca,'Using moving average low pass for anti-ailasing')
%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%% (c) %%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%
% Repeat (b) while the anti-aliasing filter is designed with the help of MATLAB function fir1()
% 
fir_x = fir1(xc_size(2)/10-1,0.5,'low');
fir_y = fir1(xc_size(1)/10-1,0.5,'low');
fir_y = fir_y';
%fir_xy = repmat(fir_x,[xc_size(1) 1]);
fir_xy = conv2(fir_x,fir_y,'full');
% figure
% plot(fir_x)
% figure
% plot(fir_y)
% figure
% imshow(fir_xy)
% fir_fft = fftshift(fft2(fir_xy));
% figure
% imshow(abs(fir_fft))
 
yc = conv2(xc,fir_xy,'same');

% downsample yc
y_sinc = yc(1:D:end, 1:D:end);

% rescale y to the original size as yc
% use the MATLAB function, imresize()
ys = imresize(y_sinc,xc_size,'bicubic');
figure
imshow(yc,[0 255])
figure
imshow(y,[0 255])
figure
imshow(ys,[0 255])
title(gca,'Using ideal low pass for anti-ailasing')
%%%%%%%%%%%%%%%%%%%%%
%%%%%%%% (d) %%%%%%%%
%%%%%%%%%%%%%%%%%%%%%
% implement your own interpolator
% first, upsample x or y
I = 2; % upsampling ratio
xu = zeros(xc_size);
yu = zeros(xc_size);
xu(1:I:end, 1:I:end) = x; % yu = ??
yu(1:I:end, 1:I:end) = y;
% then low pass filtering
% please implement the simplest low pass filter - moving average filter or try fir1()
% you may vary the kernel size of the moving average filter from 2, 4, to 8 to see the changes in reconstructed/interpolated image
N = 4;
ma = ones(N,N)/(N*N)*I*I; % filter, impulse response. Is it really need to normaized?
xs = conv2(xu,ma,'same'); % ys = ??
ys = conv2(yu,ma,'same');
figure
imshow(xs,[0 255]);
title(gca,'My interpolation MAF x')
figure
imshow(ys,[0 255]);
title(gca,'My interpolation MAF y')
%%%%%method of FIR1%%%%%%%%
xs_fir = conv2(xu,fir_xy*I*I,'same');
ys_fir = conv2(yu,fir_xy*I*I,'same');
figure
imshow(xs_fir,[0 255]);
title(gca,'My interpolation Ideal x')
figure
imshow(ys_fir,[0 255]);
title(gca,'My interpolation ideal y')
%%%%%%%%%%%%%%%%%%%
%%%%%%% (e) %%%%%%%
%%%%%%%%%%%%%%%%%%%
% % interpolate xc to the size 540 x 810
I = 3; % upsampling ratio
xu = zeros(xc_size*I);
xu(1:I:end, 1:I:end) = xc; 
% then low pass filtering
% please implement the simplest low pass filter - moving average filter or try fir1()
% you may vary the kernel size of the moving average filter from 2, 4, to 8 to see the changes in reconstructed/interpolated image
%%%%%method of FIR1%%%%%%%%
fir_x = fir1(xc_size(2)/10-1,1/3,'low');
fir_y = fir1(xc_size(1)/10-1,1/3,'low');
fir_y = fir_y';
fir_xy = conv2(fir_x,fir_y,'full');

xi = conv2(xu,fir_xy*I*I,'same');
xi = xi(1:2:end, 1:2:end);
figure
imshow(xi,[0 255]);
title(gca,'My interpolation 540x810')
