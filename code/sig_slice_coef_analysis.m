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

figure();
for i = 1:sig_slice.num
    subplot(2,2,1);hold on;
        plot(sig_slice.data{i}.c,sig_slice.data{i}.coef(1),'-ob');
        title('Offset');xlabel('Concentration');ylim([-100 100]);
    subplot(2,2,2);hold on;
        plot(sig_slice.data{i}.c,sig_slice.data{i}.coef(2),'-og');
        title('Height');xlabel('Concentration');ylim([-1000 1000]);
    subplot(2,2,3);hold on;
        plot(sig_slice.data{i}.c,sig_slice.data{i}.coef(3),'-or');
        title('Center');xlabel('Concentration');ylim([-1000 1000]);
    subplot(2,2,4);hold on;
        plot(sig_slice.data{i}.c,sig_slice.data{i}.coef(4),'-om');
        title('Slope');xlabel('Concentration');ylim([-1000 1000]);
end

