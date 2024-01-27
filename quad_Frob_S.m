function [H,g] = quad_Frob(X,F_values);
%
% Minimum value accepted for a singular value.
tol_svd = eps^5;   
%
[n,m] = size(X);
H     = zeros(n);
g     = zeros(n,1);
%
% Shift the points to the origin.
Y    = X - diag(X(:,1))*ones(n,m);
Dmax = max(sqrt(sum(Y.^2)));
Y    = Y/Dmax;
%
if ( m < (n+1)*(n+2)/2 ) 
%
%  Compute a quadratic model by minimizing the Frobenius norm of the Hessian.
%    
   b = [F_values zeros(1,n+1)]';
   A = ((Y'*Y).^2)/2;
   W = [A ones(m,1) Y';[ones(1,m);Y] zeros(n+1,n+1)];
%
%  Compute the model coefficients.
%
   if ( sum(sum(isinf(W))) > 0 ) || ( sum(sum(isnan(W))) > 0 )
       keyboard
   end
   [U,S,V]        = svd(W);
   Sdiag          = diag(S);
   indeces        = find(Sdiag < tol_svd);      
   Sdiag(indeces) = tol_svd;
   Sinv           = diag(1./Sdiag);
   lambda         = V * Sinv *U'* b;
%
%  Retrieve the model coefficients.
%                
   g = lambda(m+2:m+n+1)/Dmax;
   H = zeros(n,n);
   for j = 1:m
       H = H + lambda(j)*(Y(:,j)*Y(:,j)');
   end
   H = H/(Dmax^2);
%
else
%
%  Compute a (complete or full) quadratic model.
%
   b = F_values';
%
   phi_Q = [ ];
   for i = 1:m
       y      = Y(:,i);
       aux_H  = y*y'-1/2*diag(y.^2);
       aux    = [ ];
       for j = 1:n
           aux = [aux;aux_H(j:n,j)];
       end
       phi_Q  = [phi_Q; aux'];
   end
   W = [ones(m,1) Y' phi_Q];
%
%  Compute the coefficients of the model.
%
   [U,S,V]        = svd(W,0);
   Sdiag          = diag(S);
   indeces        = find(Sdiag < tol_svd);
   Sdiag(indeces) = tol_svd;
   Sinv           = diag(1./Sdiag);
   lambda         = V * Sinv' * U' * b;
%
%  Retrieve the model coefficients.
%
   g    = lambda(2:n+1)/Dmax;
   H    = zeros(n,n);
   cont = n+1;
   for j = 1:n
       H(j:n,j) = lambda(cont+1:cont+n-(j-1));
       cont     = cont + n - (j-1);
   end
   H = H + H' - diag(diag(H));
   H = H/(Dmax^2);
end
%
% End of quad_Frob.
