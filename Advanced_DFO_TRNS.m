%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% DFTRNS - A Derivative-free Trust-region algorithm for the unconstrained 
%          optimization of nonsmooth black-box functions.
% Copyright (C) 2019 G.Liuzzi, S.Lucidi, F.Rinaldi, L.N. Vicente:
%
% This program is free software: you can redistribute it and/or modify
% it under the terms of the GNU General Public License as published by
% the Free Software Foundation, either version 3 of the License, or
% (at your option) any later version.
%
% This program is distributed in the hope that it will be useful,
% but WITHOUT ANY WARRANTY; without even the implied warranty of
% MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
% GNU General Public License for more details.
%
% You should have received a copy of the GNU General Public License
% along with this program.  If not, see <https://www.gnu.org/licenses/>.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [f,x] = Advanced_DFO_TRNS(func_f,x_initial,omega,MAXNF,iprint)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% Purpose:
%
% Function dfo_tr_BUNDLE applies the Advanced (bundle) DFO-TRNS 
% derivative-free trust-region method to the problem:
%
%                   min f(x)
%
% where x is a real vector of dimension n.
%
% The user must provide: 
%       func_f, to evaluate the function f
%       x_initial, the initial point to start the optimizer
%                  must be a column vector
%       omega, parameter that multiplies quadratic term in TR objective
%              must be a number in [0,1]
%       MAXNF, the maximum number of allowed fun.evals
%       iprint, printing level (0 - silent; 1 - verbose)
%
% Returns:
%       f, best function value obtained
%       x, best point
%
% Functions called: func_f (application, user provided),
%                   quad_Frob (provided by the optimizer),
%                   trust (provided by MATLAB).
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% Written by G.Liuzzi, S.Lucidi, F.Rinaldi, L.N.Vicente (2019).
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
format compact;
format long;
warning('off','all');
fprintf('\n');
if(omega < 0)
    omega = 0;
elseif(omega > 1)
    omega = 1;
end
time   = clock;
n      = size(x_initial,1);
P      = haltonset(n);
isobol = 1024;
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Set tolerances, parameters and constants (not to be user-modified).
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
Delta      = 1;             % Initial trust radius.
tol_Delta  = 10^-8;         % Tolerance for stopping based on the size of the
eta0       = 0.001;         % Ratio ared/pred for considering an iterate to be
eta1       = 0.25;          % Ratio ared/pred for contracting the trust radius.
eta2       = 0.75;          % Ratio ared/pred for expanding the trust radius.
eta0_n     = 0.00000001;         % Ratio ared/pred for considering an iterate to be
eta1_n     = 0.001;          % Ratio ared/pred for contracting the trust radius.
eta2_n     = 0.1;          % Ratio ared/pred for expanding the trust radius.
theta      = 0.001;
pexpon     = 0.1;
gamma1     = 0.1;           % Contraction parameter for the trust radius.
gamma2     = 10/9;             % Expansion parameter for the trust radius.
maxY       = (n+1)*(n+2)/2; % Maximum size of the list of sampling points.
minY       = n+1;           % Minimum size of the list of sampling points.
epsilon    = 10^-5;         % Tolerance for considering a gradient small.
max_fun    = MAXNF;         % Maximum number of function evaluations.
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Set counters.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
iter         = 0; % Iteration counter.
iter_suc     = 0; % Successful iteration counter.
func_eval    = 0; % Function evaluation counter.
reduce_delta = 0;
countglob    = 0;
countsucc    = 0;
augment      = 0;
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Initialization step.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
x = x_initial;
%
% Compute the initial sample set using coordinate vectors.
nYinit = 2*n;
Y      = repmat(x,1,nYinit) + Delta * [ eye(n) -eye(n) ];
%
Y      = [ x Y ];
nY     = nYinit + 1;
%
% Evaluate f at the initial sample set.
for i=1:nY
    f_values(i) = feval(func_f,Y(:,i));
    func_eval   = func_eval + 1;
end
f = f_values(1);
fold = f;
finitial = f;
%
% Print the iteration report header.
fprintf('Iteration Report: \n\n');
if iprint > 0
    fprintf('| iter  | success |     f_value      |  tr_radius |    rho     |    |Y|  |\n');
    print_format = ['| %5d |    %2s   | %+13.8e | %+5.2e |\n'];
    fprintf(print_format, iter, '--', f, Delta );
end
%
G     = [];     % matrix whose columns are normals of hyperplanes used to build the nonsmooth model
betas = [];     % vector whose elements are rhs's of hyperplanes for the nonsmooth model
                % betas has as many elements as the columns of G 
nG    = 0;      % number of G columns

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Start the trust-region method.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
stop = 0;
while ~stop
    %
    success      = 0;
    max_red_step = 200-10*floor(iter/500);
  
    %
    %   Stop if the trust radius is too small.
    if Delta <= tol_Delta
        display('Delta<=tol Delta');
        stop = 1;
    end
    %
    %   Stop if there are too many function evaluations.
    if func_eval >= max_fun
        display('func_eval >= max_fun');
        stop = 1;
    end

    %
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %   Start the trust-region iteration.
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %
    if ~stop
        %
        %   Build a quadratic (smooth) model.
        %
        [H,g] = quad_Frob_S(Y,f_values);
        normg = norm(g);

        %   if norm of model's gradient is too small, then reduce the trust radius.
        if normg <= epsilon
            display('normg <=eps');
            reduce_delta = max_red_step;
        end
        %
        %   Minimize the model in the trust region.
       
        if (norm(H,'fro') > 10^(-5))
            [s,val] = trust(g,H,Delta);
            check_trust_sol()
        else
            s   = -(Delta/normg) * g;
            val = g'*s+0.5*s'*H*s;
        end
        fmin   = f + val;

        if abs(f - fmin)/max(1,abs(f)) < 10^-20
            % if minimization of the smooth model produces no improvement,
            % reduce the trust radius
            reduce_delta = max_red_step;
        end
        
        if (reduce_delta < max_red_step)
            xtrial    = x + s;
            ftrial    = feval(func_f,xtrial);
            func_eval = func_eval + 1;

            pred = f - fmin;
            rho  = (f - ftrial) / pred;
            
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            if rho >= eta0
                %   Set new iterate if there is success.
                success  = 1;
                iter_suc = iter_suc + 1;
                x        = xtrial;
                f        = ftrial;
                if Delta > 100*normg
                    %         Decrease trust radius.
                    if nY>=minY
                        reduce_delta = reduce_delta+1;
                    end
                else
                    %         Keep/Increase trust radius.
                    if rho >= eta2
                        Delta = min(gamma2*Delta,10^3);
                    else
                        %            Keep trust radius.
                    end
                end
                
            else
                if (abs(f - fold) < 0.000000001 * max(1.0, abs(f)) )
                    reduce_delta = reduce_delta+1;
                end
                
            end
            
        else
            countglob    = countglob+1;
            reduce_delta = 0;
            X            = repmat(x,1,nY);
            S            = Y - X;
            %
            %   Order the sample set according to the distance to the
            %   current iterate.
            index = 0;
            order_sample_set()
            S            = S(:,index);

            check_and_reset_Y()

            X         = repmat(x,1,nY);
            S         = Y - X;
            if(n > 1) 
                Ynorms    = sqrt(sum(S.^2));
            else
                Ynorms    = abs(S);
            end

            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            % construct and optimize the NonSmooth BUNDLE model
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            sol = zeros(n+1);
            construct_and_optimize_NS_model();
                
            %  Evaluate the function at the new point.
            xtrial    = x +  sol(1:n);
            ftrial    = feval(func_f,xtrial);
            func_eval = func_eval + 1;

            %fprintf('nonsmooth ====> f = %13.8e  fmod0 = %13.8e ftrial = %13.8e fsol = %13.8e\n',f,fmod0,ftrial,fsol);

            rho  = (f-ftrial) / (theta*norm(sol(1:n))^(1+pexpon));

            if rho >= eta0_n
                %display('great! nonsmooth model produced a success');

                countsucc = countsucc+1;
                success   = 1;
                augment   = 1;
                iter_suc = iter_suc + 1;
                x        = xtrial;
                f        = ftrial;
                if rho < eta1_n
                else
                    if rho >= eta2_n
                        Delta = min(gamma2*Delta,10^3);
                    end
                end
            else
                augment = 1;
                Delta = 0.9*Delta;
            end
        end
        
        % %   Updating iterate and trust-region radius.
        iter = iter + 1;
        %
        %   Order the sample set according to the distance to the
        %   current iterate.
        order_sample_set()
        
        Ynormst        = Y - repmat(xtrial,1,nY);
        if(n > 1) 
            Ynormst        = sqrt(sum(Ynormst.^2));
        else
            Ynormst        = abs(Ynormst);
        end

        if(min(Ynormst)>10^-10)
            %   Updating the sample set.
            update_sample_set()
            
            %
            %   Order the sample set according to the distance to the
            %   current iterate.
            order_sample_set()

            if ( Delta < 10^-3 & norm(s) < 10^-1 )
                fac      = 100;
                cri_minY = 3;
                while (size(Y(:,find(Ynorms<Delta*fac)),2) < max(cri_minY,3) )
                    fac = fac*2;
                end
                Y        = (Y(:,find(Ynorms<Delta*fac)));
                f_values = (f_values(find(Ynorms<Delta*fac)));
                Ynorms   = (Ynorms(find(Ynorms<Delta*fac)));
                nY       = size(Y, 2);
                G        = [];
                betas    = [];
                nG       = 0;
            end

            X = repmat(x,1,nY);
            S = Y - X;
        end
        %
        %   Print iteration report.
        print_format = ['| %5d |    %2d   | %+13.8e | %+5.2e | %+5.2e |  %5d  | %5d  | \n'];
        if iprint > 0
            fprintf(print_format,func_eval, success, f, Delta, rho, nY, nG);
        end
        %
    end
    fold=f;    
   %
end
%
time = etime(clock,time);

fprintf('# vectors used to build NS model : .......... %13d\n',nG);
fprintf('# of iters NS model was used: ............... %13d\n',countglob);
fprintf('# of iters NS model produced acceptable step: %13d\n',countsucc);
fprintf('# of function evaluations: .................. %13d\n',func_eval);
fprintf('Final TR radius: ............................ %13.6e\n',Delta);
fprintf('Initial function value: ..................... %13.6e\n',finitial);
fprintf('Final   function value: ..................... %13.6e\n',f);

    function construct_and_optimize_NS_model()
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % begin construction of nonsmooth model (max of hyperplanes)
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        nC      = 10;
        Gt      = -1 + 2*P([isobol:isobol+nC-1],:)';
        isobol  = isobol + nC;
        if(n > 1)
            Gtnorms = sqrt(sum(Gt.^2));
        else
            Gtnorms = abs(Gt);
        end
        Gt      = Gt./repmat(Gtnorms,n,1);

        G       = [G Gt];
        nG      = size(G,2);
        betas   = max(0.0,max(-(repmat(f_values,nG,1) - repmat(f,nG,nY) - G'*S - 0.00001*repmat(Ynorms,nG,1)),[],2))';
        tildebetas = betas;
        tildebetas(1:end-nC) = tildebetas(1:end-nC) + 1.0*sqrt(Delta);

        %display(['min betas = ' num2str(min(tildebetas))]);
        %display(['max betas = ' num2str(max(tildebetas))]);

        if (norm(H,'fro') > 10^(-5))
            lambda=eig(H);

            if( (min(lambda)>0.000001) && (max(lambda)<100000) )
                Q = [H zeros(n,1);zeros(1,n+1)];
            else
                Q = [0.5*min(100,1/sqrt(Delta))*eye(n,n) zeros(n,1);zeros(1,n+1)];
            end
        else
                Q = [0.5*min(100,1/sqrt(Delta))*eye(n,n) zeros(n,1);zeros(1,n+1)];
        end

        Q          = omega*Q;
        flin       = [zeros(n,1);1];
        A          = [G' -ones(nG,1)];
        b          = -repmat(f,nG,1) + tildebetas';
        fmod0      = max(repmat(f,nG,1) -  tildebetas');
        fun        = @(x)quadobj(x,Q,flin,0);
        nonlconstr = @(x)quadconstr(x,Delta);
        x0         = [x;fmod0];

        options = optimoptions(@fmincon,'Algorithm','active-set',...
            'GradObj','on','GradConstr','on',...
            'HessFcn',@(x,lambda)quadhess(x,lambda,Q),'Display',...
            'off', 'MaxFunEvals', 1000,'MaxIter',1000);

        [sol,fsol,eflag,output,lambda] = fmincon(fun,x0,A,b,[],[],[],[],nonlconstr,options);

        if norm(G*lambda.ineqlin,2) < 1.e-1*sqrt(Delta)
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            % begin construction of nonsmooth model (max of hyperplanes)
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            G = G(:,end);
            nG = 1;

            betas   = max(0.0,max(-(repmat(f_values,nG,1) - repmat(f,nG,nY) - G'*S - 0.00001*repmat(Ynorms,nG,1)),[],2))';
            tildebetas = betas;

            %display(['min betas = ' num2str(min(tildebetas))]);
            %display(['max betas = ' num2str(max(tildebetas))]);

            if (norm(H,'fro') > 10^(-5))
                lambda=eig(H);

                if( (min(lambda)>0.000001) && (max(lambda)<100000) )
                    Q = [H zeros(n,1);zeros(1,n+1)];
                else
                    Q = [0.5*min(100,1/sqrt(Delta))*eye(n,n) zeros(n,1);zeros(1,n+1)];
                end
            else
                    Q = [0.5*min(100,1/sqrt(Delta))*eye(n,n) zeros(n,1);zeros(1,n+1)];
            end

            Q          = omega*Q;
            flin       = [zeros(n,1);1];
            A          = [G' -ones(nG,1)];
            b          = -repmat(f,nG,1) + tildebetas';
            fmod0      = max(repmat(f,nG,1) -  tildebetas');
            fun        = @(x)quadobj(x,Q,flin,0);
            nonlconstr = @(x)quadconstr(x,Delta);
            x0         = [x;fmod0];

            options = optimoptions(@fmincon,'Algorithm','active-set',...
                'GradObj','on','GradConstr','on',...
                'HessFcn',@(x,lambda)quadhess(x,lambda,Q),'Display',...
                'off', 'MaxFunEvals', 1000,'MaxIter',1000); %, ...

            [sol,fsol,eflag,output,lambda] = fmincon(fun,x0,A,b,[],[],[],[],nonlconstr,options);
        end        
    end
    function check_and_reset_Y()
        raggio       = min(Delta,10.0);
        indices      = find((Ynorms <= raggio) & (Ynorms > 0.0000001));

        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %  refresh points if empty set of indices is obtained
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

        if(isempty(indices)) || (length(indices) <= 1)
            npmax    = max(3,n/3);
            Y        = x;
            f_values = f;
            nY       = 1; 
            for r=1:npmax
                nY        = nY + 1;
                d         = -1 + 2*P(isobol,:)';
                isobol    = isobol+1;
                d         = d./norm(d)* 0.5*raggio;
                xtrial    = x + d;
                ftrial    = feval(func_f,xtrial);
                func_eval = func_eval + 1;

                Y(:,nY)      = xtrial;
                f_values(nY) = ftrial;
            end
        end
    end
    function check_trust_sol()
        if norm(s) > Delta+1.e-10
            if g'*H*g > 0
                alfastar = (g'*g)/(g'*H*g);
                if norm(alfastar*g) <= Delta
                    s = -alfastar * g;
                    val = g'*s+0.5*s'*H*s;
                else
                    alfastar = Delta/normg;
                    %keyboard
                    s   = -alfastar * g;
                    val = g'*s+0.5*s'*H*s;
                end
            else
                    alfastar = Delta/normg;
                    %keyboard
                    s   = -alfastar * g;
                    val = g'*s+0.5*s'*H*s;
            end
        end
    end
    function update_sample_set()
        if success || augment
            if nY < maxY
                nY = nY + 1;
            end
            Y(:,nY)      = xtrial;
            Ynorms(nY)   = norm(xtrial-x);
            f_values(nY) = ftrial;
            %
        else
            if (nY == maxY )
                if norm(xtrial - x) <= Ynorms(nY)
                    Y(:,nY)      = xtrial;
                    f_values(nY) = ftrial;
                end
            else
                nY           = nY + 1;
                Y(:,nY)      = xtrial;
                f_values(nY) = ftrial;
            end
        end
    end
    function order_sample_set()
        Ynorms         = Y - repmat(x,1,nY);
        if(n > 1)
            Ynorms         = sqrt(sum(Ynorms.^2));
        else
            Ynorms         = abs(Ynorms);
        end
        [Ynorms,index] = sort(Ynorms);
        Y              = Y(:,index);
        f_values       = f_values(index);
    end
%
% End of Advanced_DFO_TRNS.
%
end

function [y,grady] = quadobj(x,Q,f,c)
    y = 0.5 * 1/2*x'*Q*x + f'*x + c;
    if nargout > 1
        grady = 0.5 * Q*x + f;
    end
end

function [y,yeq,grady,gradyeq] = quadconstr(xv,d)
    n = length(xv);
    x = xv(1:(n-1));
    y = 1/2*x'*x - d^2/2;
    yeq = [];

    if nargout > 2
        grady = [x;0.0];
    end
    gradyeq = [];
end

function hess = quadhess(x,lambda,Q)
    [n,m] = size(Q);
    hess = Q;
    Id = eye(n,n);
    Id(n,n) = 0.0;
    hess = hess + lambda.ineqnonlin(1)*Id;
end
