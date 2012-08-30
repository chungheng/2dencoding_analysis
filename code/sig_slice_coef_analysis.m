%% Analysis the coefficient of sigmoidal regression for c-slice
%
%
% Synopsis : the sigmoidal regression result is in ../data/sig_slice.mat
% Author   : Chung-Heng Yeh <chyeh@ee.columbia.edu>
% Note     : The experimental is provides by Yevgeniy Slutskiy

% [1] A. J. Kim, A. A. Lazar and Y. B. Slutskiy, System Identification of 
% Drosophila Olfactory Sensory Neurons, Journal of Computational 

clc;clear all;close all;

load ../data/sig_slice.mat;

f = @(x) x;

concen = cellfun(@(x) x.c, sig_slice.data);
offset = cellfun(@(x) x.coef(1), sig_slice.data);
height = cellfun(@(x) x.coef(2), sig_slice.data);
center = cellfun(@(x) x.coef(3), sig_slice.data);
width  = cellfun(@(x) x.coef(4), sig_slice.data);

figure();
subplot(2,3,1);
    plot(concen,offset,'ob');grid on;
    title('Offset');xlabel('Concentration');ylim([-50 100]);
subplot(2,3,2);
    plot(concen,height,'og');grid on;
    title('Height');xlabel('Concentration');ylim([0 400]);
subplot(2,3,4);
    plot(concen,center,'or');grid on;
    title('Center');xlabel('Concentration');ylim([0 500]);
subplot(2,3,5);
    plot(concen,width,'om');grid on;
    title('Width');xlabel('Concentration');ylim([0 400]);
subplot(2,3,[3 6]);
    plot(concen,height./width,'ok');grid on;
    title('Gain');xlabel('Concentration');ylim([0 2.5]);
