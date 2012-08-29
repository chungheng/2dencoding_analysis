%% Gain Control using 2D Encoding data in [1]
%
% Synopsis : Implement IAF neuron with gain control
% Author   : Chung-Heng Yeh <chyeh@ee.columbia.edu>
% Note     : The experimental is provides by Yevgeniy Slutskiy


% Concentration Range: 19~111
% 
clc; close all;clear all;
load 2D_slope_points.mat
idx = find(abs(first_slope) < 0.05);
fst_coef = pinv([concentration(idx)' ones(length(concentration(idx)),1)],1e-8)*first_slope(idx)';
idx = find( concentration < 75 & abs(second_slope) <0.5);
sec_coef = pinv([concentration(idx)' ones(length(concentration(idx)),1)],1e-8)*second_slope(idx)';
idx = find( concentration < 75 );
bp_x_coef = pinv([concentration(idx)' ones(length(concentration(idx)),1)],1e-8)*break_point(1,idx)';
bp_y_coef = pinv([concentration(idx)' ones(length(concentration(idx)),1)],1e-8)*break_point(2,idx)';


fst_reg = fst_coef(1)*concentration + fst_coef(2);
sec_reg = sec_coef(1)*concentration + sec_coef(2);
x_reg   = bp_x_coef(1)*concentration + bp_x_coef(2);
y_reg   = bp_y_coef(1)*concentration + bp_y_coef(2);

figure()
idx = find(abs(first_slope) < 1);
subplot(3,1,1);plot(concentration(idx),first_slope(idx),'-+b',concentration,fst_reg,'-k');
title('Slope of Segments');xlabel('Concentration');ylabel('Slope');
idx = find(abs(second_slope) < 1);
subplot(3,1,2);plot(concentration(idx),second_slope(idx),'-xR',concentration,sec_reg,'-k');
title('Slope of Segments');xlabel('Concentration');ylabel('Slope');
idx = find( concentration < 75 );
subplot(3,1,3);plot3(concentration(idx),break_point(1,idx),break_point(2,idx),'-og',...
                     concentration(idx),x_reg(idx),y_reg(idx),'-k');
title('Breakpoint');xlabel('Concentration');ylabel('Rate-of-change');grid on;


save iaf_gain_control.mat fst_coef sec_coef bp_x_coef bp_y_coef

%% 
% Test iaf_encode_gc
clc; clear all; close all;
load 2009_10_17_00_a_samples_040_000.mat

color_palette = 'rbgmyk';
figure();
for idx = 45:50;

color = color_palette(mod(idx,6)+1);

u = samples{idx};
t = sample_times{idx}*1e-4;

ts = timeseries(t',u(:,1));

dt = 1e-5;
res_ts=resample(ts,t(1):dt:t(end));


iter_num = 1;
[spk,v ,r] = iaf_encode_gc(res_ts.data,res_ts.time,3e-2);
[t_psth, psth] = compute_psth(res_ts.time,spk);
for i = 1:iter_num-1
    spk = iaf_encode_gc(res_ts.data,res_ts.time,3e-2);
    [t_psth, psth_temp] = compute_psth(res_ts.time,spk);
    psth = psth + psth_temp;
end
psth = psth / iter_num;


subplot(2,1,1);hold on;grid on;
plot(t,u(:,1),color);
xlabel('time, sec');ylabel('Concentration, ppm');
subplot(2,1,2);hold on;grid on;
plot(res_ts.time,r,color);
title('PSTH')
xlabel('time, sec');ylabel('Frequency, Hz');

end
