%----------------------------
% Function l1hilb
%----------------------------
function y = l1hilbert(x)
    n=length(x);
    i = [1:n]'; I = repmat(i,1,n);
    j = [1:n];  J = repmat(j,n,1);
    X = repmat(x',n,1);

    f = sum(X./(I+J-1),2);

    y = max(abs(f));
end
