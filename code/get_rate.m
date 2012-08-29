function rate = get_rate(s,dur,dt,varargin)


% Handle the optional input parameters, e.g., Calc_MSE_N     
 p = inputParser;
 addParamValue(p,'WinLen',200e-3,@isnumeric);
 p.KeepUnmatched = true;

 parse(p,varargin{:});
 UnmatchedParam = fieldnames(p.Unmatched);
 if ~isempty(UnmatchedParam)
    error(['"',UnmatchedParam{1},'" is not a valid paprameter. ' ...
               'Please use "WinLen" to specify window length']);
 end

win = p.Results.WinLen;
%s=cumsum(s);
rate=zeros(1,ceil(dur/dt+1));
for i=1:ceil(dur/dt)
    rate(i)=sum((s>=(i*dt-win) & s<= (i*dt)))/win;
end