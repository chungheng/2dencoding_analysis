%IAF_ENCODE_GC 
function [s, v, r]  = iaf_encode_gc(u, t, noise)

%load iaf_gain_control.mat
load 2D_slope_points.mat
s = zeros(1, length(t));        % initialize the spikes list
v = zeros(1, length(t));        % initialize the voltage trace
r = zeros(1, length(t));        % initialize the voltage trace

e = 0;                          % initialize the intrgration error term
d = 0.07;
k = 1;
for i=2:length(t)
    dt = t(i) - t(i-1);
    % compute the membrane voltage
    du = (u(i) - u(i-1)) / dt;
    idx = max(1,round((u(i)-19)/0.5));
    if du < break_point(1,idx)
        rate = (du-break_point(1,idx))*first_slope(idx) + break_point(2,idx);
    else
        rate = (du-break_point(1,idx))*second_slope(idx) + break_point(2,idx);
    end
    %{

    bp_x = u(i)*bp_x_coef(1) + bp_x_coef(2); 
    bp_y = u(i)*bp_y_coef(1) + bp_y_coef(2); 
    fst_slope = u(i)*fst_coef(1) + fst_coef(2);
    sec_slope = u(i)*sec_coef(1) + sec_coef(2);
    if du > bp_x,
        rate = (du-bp_x)*sec_slope + bp_y;
    else
        rate = (du-bp_x)*fst_slope + bp_y;
    end
    %}
    
    if u(i) < 19.5
        rate = 0.0;
    end
    r(i) = rate;
    b = k*d*rate - u(i);
    v(i) = e + v(i-1) + ( b + 0.5*(u(i)+u(i-1)) )*dt/k;     % 6/17/2012
    
    d_temp = d + randn()*noise^2;
    % if the voltage is above the threshold d, generate a spike
    if v(i) >= d_temp                
        s(i) = 1;               % generate a spike
        v(i) = v(i)-d_temp ;    % reset the voltage
        e = v(i);               % keep track of the integration error           
    else
        e = 0;
    end
end

end % end of function
