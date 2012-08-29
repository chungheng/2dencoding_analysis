%% Test of 2D Encoding data in [1]
%
%
% Synopsis : Play around the data
% Author   : Chung-Heng Yeh <chyeh@ee.columbia.edu>
% Note     : The experimental is provides by Yevgeniy Slutskiy

% [1] A. J. Kim, A. A. Lazar and Y. B. Slutskiy, System Identification of 
% Drosophila Olfactory Sensory Neurons, Journal of Computational 
% Neuroscience, Vol. 30, No. 1, pp. 143?161, 2011.

%% Reorganize data for easier usage
clc; clear all;close all;
load ../data/2009_10_17_00_a_samples_040_000.mat
data = cell(1,length(samples));
% Remove negative data 
for i = 1:length(samples)
    data{i} = struct('t',sample_times{i}*1e-4,...
                     'c',samples{i}(:,1),...
                     'dc',samples{i}(:,2),...
                     'psth',samples{i}(:,3));
end

sample = struct( 'num', length(samples), 'data' , {data});
save( '../data/sample.mat','sample');


%% Updample the original data
close all; clear all; clc;
load ../data/2009_10_17_00_a_samples_040_000.mat

dt = 1e-4;
sample_num = length( samples );
data = cell(1,sample_num);

for i = 1:sample_num
    s = ( spike_times{i} - t(1) ) * dt;
    rate = get_rate( s, (t(end)-t(1))*dt, dt );
    data{i} = struct('c', odor{i}(1,:),...
                     'dc',odor{i}(2,:),...
                     'psth',rate);
end

upSample = struct('num',sample_num,'t',t*dt,'data',{data});
save upSample upSample;
clear all;


%% Downsample the upsampled data
clc; clear all; close all;

load ../data/upSample.mat
data = cell(1,upSample.num);
% Remove negative data 
for i = 1:upSample.num
    idx = find( upSample.data{i}.c > 0.0 );
    data{i} = struct('t',downsample(upSample.t(idx),100 ),...
                     'c',downsample(upSample.data{i}.c(idx),100 ),...
                     'dc',downsample(upSample.data{i}.dc(idx),100 ),...
                     'psth',downsample(upSample.data{i}.psth(idx),100 ));
end
downSample = struct('num',upSample.num,'data',{data});

save('../data/downSample.mat','downSample');

%% Plot the PSTH of the raw data, upsample
clc; clear all;close all;
load ../data/2009_10_17_00_a_samples_040_000.mat
load ../data/downSample.mat
load ../data/upSample.mat
idx = 50;
figure();
subplot(3,1,1);plot(sample_times{idx},samples{idx}(:,3));title('raw data');
subplot(3,1,2);plot(downSample.data{idx}.t,downSample.data{idx}.psth);title('Downsample');
subplot(3,1,3);plot(upSample.t,upSample.data{idx}.psth);title('Upsample');
xlim([8.5 11.5]);

%% Plot 2D and 3D waveform of the raw data
clc; clear all; close all;
load ../data/2009_10_17_00_a_samples_040_000.mat

baseColor = [[0,0,0];[0,1,0]];
range = 1:110;

figure()

for i = range
    hold on;
    plot(1:size(samples{i},1),samples{i}(:,1),...
         'Color',[(i-range(1))/length(range) 1-(i-range(1))/length(range)]*baseColor);
end
grid on;
figure()
for i = range
    hold on;
    plot3(samples{i}(:,1),samples{i}(:,2),samples{i}(:,3),...
         'Color',[(i-range(1))/length(range) 1-(i-range(1))/length(range)]*baseColor);
end
grid on;

%% 
% Piecewise Linear Regression for data points on the slice of fixed 
% concentration
clc; clear all; close all;
load ../data/2009_10_17_00_a_samples_040_000.mat

for C = 5:1:200
    baseColor = [[0,0,0];[0,1,0]];
    range = [C-0.5 C+0.5];
    x = [];
    y = [];
    for i = 1:length(samples)
    
        idx = find( samples{i}(:,1) > range(1) & samples{i}(:,1) <= range(2));
        x = [x; samples{i}(idx,2)];
        y = [y; samples{i}(idx,3)];
    end

    slm = slmengine(x,y,'degree',1,'plot','on', 'knots',4,'interiorknots','free');
    title(['Concentration: ', num2str(bias)]);
    print(gcf,'-dpng','-r300',['../pic/pic_concentration_slice_3/', num2str(bias), '.png']);
    close gcf;
end

%%
% Piecewise Linear Regression for data points on the slice of fixed 
% Rate-of-Change
clc; clear all; close all;
load ../data/2009_10_17_00_a_samples_040_000.mat

dc_gap = 20;
for bias = -900:dc_gap:1000
    baseColor = [[0,0,0];[0,1,0]];
    range = [bias-1 bias+1];
    x = [];
    y = [];
    for i = 1:length(samples)
    
        idx = find( samples{i}(:,2) > range(1) & samples{i}(:,2) <= range(2));
        x = [x; samples{i}(idx,1)];
        y = [y; samples{i}(idx,3)];

    end

    slm = slmengine(x,y,'degree',1,'plot','on', 'knots',5,'interiorknots','free');
    title(['Concentration: ', num2str(bias)]);
    print(gcf,'-dpng','-r300',['../pic/pic_gradient_slice_2/', num2str(bias), '.png']);
    close gcf;
end

%%
% Plot slopes of first and second segment and the trajectory of the first
% break point, also export result to mat file.

clc; clear all; close all;
load ../data/2009_10_17_00_a_samples_040_000.mat

knot_num = 5;
gap = 1;
concentration = 5:gap:180;
first_slope   = zeros(1,length(concentration));
second_slope  = zeros(1,length(concentration));
break_point   = zeros(2,length(concentration));

data = cell(1,length(concentration));
for j = 1:length(concentration)
    C = concentration(j);
    range = [C-gap/2 C+gap/2];
    x = [];
    y = [];
    for j = 1:length(samples)
        idx = find( samples{j}(:,1) > range(1) & samples{j}(:,1) <= range(2));
        x = [x; samples{j}(idx,2)];
        y = [y; samples{j}(idx,3)];
    end

    slm = slmengine(x,y,'degree',1,'knots',knot_num,'interiorknots','free');
    first_slope(i)   = (slm.knots(2)-slm.knots(1))/(slm.coef(2)-slm.coef(1));
    second_slope(i)  = (slm.knots(3)-slm.knots(2))/(slm.coef(3)-slm.coef(2));

    break_point(:,i) = [slm.knots(2); slm.coef(2)];
    data{i} = struct('c',C,'breakpoints',[slm.knots slm.coef]);
end
% Export mat file 
piecelin_slice = struct('knot_num',knot_num,'num',length(concentration),...
                        'c_gap',gap,'data',{data});
save('../data/piecelin_slice.mat','piecelin_slice');
figure()
% slope of the 1st segment
subplot(3,1,1);plot(concentration,first_slope,'-+b');
title('Slope of Segments');xlabel('Concentration');ylabel('Slope');
% slope of the 2nd segment
subplot(3,1,2);plot(concentration,second_slope,'-xR');
title('Slope of Segments');xlabel('Concentration');ylabel('Slope');
% Breakpoint
subplot(3,1,3);plot3(concentration,break_point(1,:),break_point(2,:),'-ok');
title('Breakpoint');xlabel('Concentration');ylabel('Rate-of-change');
zlabel('Frequency');grid on;
print(gcf,'-dpng','-r600','../pic/slope_1st_2nd_breakpoint.png');


%%
figure()
for i = 1:length(concentration)
    hold on;
    plot3(regressor{i}.c*ones(num_knots), ...
          regressor{i}.breakpoints(:,1), ...
          regressor{i}.breakpoints(:,2), ...
          '-o','Color',ColorSet(i,:),'MarkerFaceColor','k');
end
grid on; xlabel('Concentration'); ylabel('rate-of-change');
zlabel('Frequency');
title('Piecewise Linear Regression of 2D Encoding Manifold');
xlim([0 200]);
%%
figure()
idx = find(abs(first_slope) < 1);
subplot(3,1,1);plot(concentration(idx),first_slope(idx),'-+b');
title('Slope of Segments');xlabel('Concentration');ylabel('Slope');
idx = find(abs(second_slope) < 1);
subplot(3,1,2);plot(concentration(idx),second_slope(idx),'-xR');
title('Slope of Segments');xlabel('Concentration');ylabel('Slope');

subplot(3,1,3);plot3(concentration,break_point(1,:),break_point(2,:),'-ok');
xlim([19 80]);
title('Breakpoint');xlabel('Concentration');ylabel('Rate-of-change');
zlabel('Frequency');grid on;


%% 
% Two planes fitting using CVX;
% The result is not good.
clc; close all; clear all;
load 2009_10_17_00_a_samples_040_000.mat

c    = [];
dc   = [];
psth = [];
c_threshold = 100;
dc_threshold = 200;

for i=1:length(samples)
    idx = find(abs(samples{i}(:,2)) < dc_threshold & samples{i}(:,1)< c_threshold);
    c = [c; samples{i}(idx,1)];
    dc = [dc; samples{i}(idx,2)];
    psth = [psth; samples{i}(idx,3)];

end

display('Finished packaging data.')
x = [c dc ones(size(dc))];

lambda = 1333;
n = size(x,1);
cvx_begin
    variable z(n);
    variable A1(3);
    variable A2(3);
    minimize norm(z-psth,2)+lambda*norm(A1,2)+lambda*norm(A1,2);
    subject to
        x*A1 <= z;
        x*A2 <= z;
cvx_end
        
pre = max([x*A1 x*A2],2);
plot3(c,dc,psth,'b',c,dc,pre,'g')
xlabel('Concentration');ylabel('Rat-of-change');zlabel('Frequency');
legend('Raw data','Predicted');


%% 


%%
figure('Position',[0 0 1500 1200]);


%%
% Fit raw data with sigmoidal function for c-slice with nlinfit and fit

clc; clear all; close all;
load ../data/sample.mat

gap = 1;
concentration = 5:gap:240;

ColorSet = varycolor(length(concentration));
f = @(p,x) p(1) + p(2) ./ (1 + exp(-(x-p(3))/p(4)));

data = cell(1,length(concentration));
dc = -1500:1500;
figure();
for i = 1:length(concentration)
    C = concentration(i);
    range = [C-gap/2 C+gap/2];
    x = [];
    y = [];
    for j = 1:sample.num
        idx = find( sample.data{j}.c > range(1) & sample.data{j}.c  <= range(2));
        x = [x; sample.data{j}.dc(idx)];
        y = [y; sample.data{j}.psth(idx)];
    end
    p = nlinfit(x,y,f,[0 20 50 5]);
    
    x = sort(x);
    line(C*ones(size(x)),x,f(p,x),'color',ColorSet(i,:));
    
    z = f(p,dc);
    %hold on; plot(dc,z,'color',ColorSet(ji,:));
    data{i} =  struct('range',[x(1) x(end)],'c',C,'coef',p);
   
end
xlim([0 250]);grid on;
xlabel('Concentration');ylabel('Rate-of-change');
zlabel('Frequency');grid on;view(50, 40);
title({'Sigmoidal Regression','Slice of fixed concentration'});



sig_slice = struct('data',{data},'num',length(concentration));
save('../data/sig_slice.mat','sig_slice');


%% Fit raw data with generalized Sigmoidal function
clc; clear all; close all;
load ../data/sample.mat

f  = @(p,x) x(:,1)*p(2) + p(2) + ( x(:,1)*p(3) + p(4) ) ./ (1 + exp( -(x(:,2)-( x(:,1)*p(5) + p(6) ))./( x(:,1)*p(7) + p(8) )));

x = [];
y = [];
for i = 1:sample.num
    x = [x; sample.data{i}.c sample.data{i}.dc];
    y = [y; sample.data{i}.psth];
end
p = nlinfit(x,y,f,[0 0 0 200 0 10 0 600]);
sig_surface = struct('coef',p);
save('../data/sig_surface.mat','sig_surface');
[MX MY] = meshgrid(0:250,-2000:1500);
Z = f2(p,MX,MY);
figure();
surf(MX,MY,Z,'EdgeColor','none');
xlim([0 200]);grid on;view(50, 40);

%% 
% Plot piecewise linear regression for c-slice, sigmoidal regression for 
% c-slice, and generalized sigmoidal surface regression
clc; clear all; close all;

load ../data/piecelin_slice.mat
load ../data/sig_slice.mat
load ../data/sig_surface.mat



figure('Name','2D_Manifold_Regression');
% 3D plot 


subplot(2,3,1);
    colorset = varycolor(piecelin_slice.num);
    for i = 1:piecelin_slice.num
        hold on;
        plot3(piecelin_slice.data{i}.c*ones(piecelin_slice.knot_num),...
              piecelin_slice.data{i}.breakpoints(:,1),...
              piecelin_slice.data{i}.breakpoints(:,2),...
              'Color',colorset(i,:));
    end
    grid on;view(50, 40);title({'Piecewise Linear Regression of c-Slice'});
    xlabel('Concentration');ylabel('Rate-of-Change');zlabel('Frequency');
    xlim([0 200]);ylim([-2000 2500]);
    
subplot(2,3,2);
    colorset = varycolor(sig_slice.num);
    f = @(p,x) p(1) + p(2) ./ (1 + exp(-(x-p(3))/p(4)));
    for i = 1:sig_slice.num
        hold on;
        dc = sig_slice.data{i}.range(1):sig_slice.data{i}.range(2);
        plot3(sig_slice.data{i}.c*ones(size(dc)),dc,...
              f(sig_slice.data{i}.coef,dc),...
              'Color',colorset(i,:));
    end
    grid on;view(50, 40);title({'Piecewise Linear Regression of c-Slice'});
    xlabel('Concentration');ylabel('Rate-of-Change');zlabel('Frequency');
    xlim([0 200]);ylim([-2000 2500]);
    
subplot(2,3,3);
    f = @(p,x,y) x*p(2) + p(2) + ( x*p(3) + p(4) ) ./ (1 + exp( -(y-( x*p(5) + p(6) ))./( x*p(7) + p(8) )));
    [MX MY] = meshgrid(0:250,-2000:1500);
    Z = f(sig_surface.coef,MX,MY);
    subplot(2,3,3);
    surf(MX,MY,Z,'EdgeColor','none');
    grid on;view(50, 40);title({'Sigmoidal Surface Regression'});
    xlabel('Concentration');ylabel('Rate-of-Change');zlabel('Frequency');
    xlim([0 200]);ylim([-2000 2500]);
    
subplot(2,3,4);
    xlabel('Rate-of-Change');ylabel('Frequency');grid on;
    xlim([-1500 1500]);ylim([0 300]);
    colorset = varycolor(piecelin_slice.num);
    for i = 1:piecelin_slice.num
        hold on;
        plot(piecelin_slice.data{i}.breakpoints(:,1),...
             piecelin_slice.data{i}.breakpoints(:,2),...
             '-','Color',colorset(i,:));
    end

subplot(2,3,5);
    xlabel('Rate-of-Change');ylabel('Frequency');grid on;
    xlim([-1500 1500]);ylim([0 300]);
    colorset = varycolor(sig_slice.num);
    f = @(p,x) p(1) + p(2) ./ (1 + exp(-(x-p(3))/p(4)));
    dc = -1500:1500;
    for i = 1:piecelin_slice.num
        hold on;
        plot(dc,f(sig_slice.data{i}.coef,dc),...
             '-','Color',colorset(i,:));
    end
    xlabel('Rate-of-change');
    ylabel('Frequency');grid on;
    xlim([-1500 1500]);ylim([0 300]);
    
subplot(2,3,6);
f = @(p,x,y) x*p(2) + p(2) + ( x*p(3) + p(4) ) ./ (1 + exp( -(y-( x*p(5) + p(6) ))./( x*p(7) + p(8) )));
    
    xlabel('Rate-of-Change');ylabel('Frequency');grid on;
    xlim([-1500 1500]);ylim([0 300]);
    dc = (-1500:1500)';
    gap = 0.5;
    concentration = 5:gap:250;
    ColorSet = varycolor(length(concentration));
    for j = 1:length(concentration)
        c = concentration(j)*ones(size(dc));
        z = f(sig_surface.coef,c,dc);
        subplot(2,3,6);
        hold on; plot(dc,z,'color',ColorSet(j,:));
    end
