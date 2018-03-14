function [rho,P,p,lambda,eta] = FixedStepMethod(mu,L,alpha,beta,gamma,rho,tol,solver,PD_cons)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Finds a quadratic Lyapunov function (if one exists) for a first-order
% iterative fixed-step method applied to a smooth strongly convex function.
% 
% Note: This function requires YALMIP
% 
% Inputs:
%   mu      -  strong convexity parameter
%   L       -  smoothness parameter (Lipschitz constant of gradient)
%   alpha   -  step-sizes for gradients in update equation
%   beta    -  step-sizes for iterates in update equation
%   gamma   -  step-sizes for gradients in output equation
%   rho     -  either convergence rate to test, or upper and lower bounds for bisection
%   tol     -  tolerance for bisection
%   solver  -  SDP solver
%   PD_cons -  0/1 constraint on P>0 and p>0 (default is 0)
%
% Outputs:
%   P,p,lambda  -  Lyapunov function parameters
%   eta         -  slack term coefficients in decrease of Lyapunov function
%   rho         -  convergence rate
% 
% Usage:
%   []                   = FixedStepMethod(mu,L)
%   [  ~,P,p,lambda,eta] = FixedStepMethod(mu,L,alpha,beta,gamma,rho)
%   [rho,P,p,lambda,eta] = FixedStepMethod(mu,L,alpha,beta,gamma)
%   [rho,P,p,lambda,eta] = FixedStepMethod(mu,L,alpha,beta,gamma,[rho_lower,rho_upper],tol)
% 
% To see examples, type:  mu=1; L=10; FixedStepMethod(mu,L);
% 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

if nargin == 2
    
    fprintf('%13s   %6s\n','Method','Rate');
    
    % Gradient Method
    rho = FixedStepMethod(mu,L,1/L,1,1);
    fprintf('%13s : %6.4f\n','GM (1/L)',rho);
    
    rho = FixedStepMethod(mu,L,2/(L+mu),1,1);
    fprintf('%13s : %6.4f\n','GM (2/(L+mu))',rho);

    % Heavy Ball Method
    alpha = 4/(sqrt(L)+sqrt(mu))^2;
    beta  = ((sqrt(L)-sqrt(mu))/(sqrt(L)+sqrt(mu)))^2;
    gamma = 0;
    rho   = FixedStepMethod(mu,L,alpha,[1+beta,-beta],[1+gamma,-gamma]);
    fprintf('%13s : %6.4f\n','HBM',rho);
    
    % Fast Gradient Method
    alpha = 1/L;
    beta  = (sqrt(L)-sqrt(mu))/(sqrt(L)+sqrt(mu));
    gamma = beta;
    rho   = FixedStepMethod(mu,L,alpha,[1+beta,-beta],[1+gamma,-gamma]);
    fprintf('%13s : %6.4f\n','FGM',rho);
    
    % Triple Momentum Method
    RHO   = 1-sqrt(mu/L);
    alpha = (1+RHO)/L;
    beta  = RHO^2/(2-RHO);
    gamma = RHO^2/((1+RHO)*(2-RHO));
    rho   = FixedStepMethod(mu,L,alpha,[1+beta,-beta],[1+gamma,-gamma]);
    fprintf('%13s : %6.4f\n','TMM',rho);
    
    rho    = [];
    P      = [];
    p      = [];
    lambda = [];
    eta    = [];
    return;
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% CHECK INPUTS
N = length(beta)-1;

assert(N==length(gamma)-1,'beta and gamma must both be length N');
assert(N>=0,'Need N>=0');

if nargin<8 || isempty(solver)
    solver = 'mosek';
end
if nargin<9 || isempty(PD_cons)
    PD_cons = 0;
end

if nargin>=6 && length(rho)==1
    
    % If given rho, solve the SDP to find a Lyapunov function (if possible)
    [~,P,p,lambda,eta] = SolveSDP(mu,L,alpha,beta,gamma,rho,N,solver,PD_cons);
    
else
    % Bounds to use for bisection
    %  - Use [rho_lower, rho_upper] if specified
    %  - Otherwise use [0,1]
    if nargin>=6 && length(rho)==2
        rho_lower = rho(1);
        rho_upper = rho(2);
    else
        rho_lower = 0;
        rho_upper = 1;
    end
    
    % Default tolerance for bisection
    if nargin<7 || isempty(tol)
        tol = 1e-4;
    end
    
    % Perform bisection on rho to find the minimum rho for which the SDP is
    % feasible
    while rho_upper-rho_lower >= tol
        
        rho = (rho_lower + rho_upper)/2;
        
        flag = SolveSDP(mu,L,alpha,beta,gamma,rho,N,solver,PD_cons);
        
        % Update bounds
        if flag
            rho_upper = rho;
        else
            rho_lower = rho;
        end
    end
    
    % Return the solution at the minimum rho for which the problem is feasible
    rho = rho_upper;
    [~,P,p,lambda,eta] = SolveSDP(mu,L,alpha,beta,gamma,rho,N,solver,PD_cons);
end

end


function [flag,P,p,lambda,eta] = SolveSDP(mu,L,alpha,beta,gamma,rho,N,solver,PD_cons)

% SDP variables
P   = sdpvar(2*(N+1));
p   = sdpvar(N+1,1);
eta = sdpvar(N+3,N+3,'full');

cons = (eta>=0);  % Constraints

if ~PD_cons % if we do not constrain P>0 and p>0 (mostly for Figure 2)
    lambda = sdpvar(N+2,N+2,'full');
    cons = cons + (lambda>=0);
else
    lambda = zeros(N+2,N+2);
end
 
[v,V, ~, ~,  m_N,  M_N] = SetupSDP(mu,L,alpha,beta,gamma,rho,N,P,p,N);
[~,~,dv,dV,m_Np1,M_Np1] = SetupSDP(mu,L,alpha,beta,gamma,rho,N,P,p,N+1);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 5) SEMIDEFINITE PROGRAM

% V must be positive definite
for i = 1:N+2
    for j = 1:N+2
        v = v - lambda(i,j)*m_N{i,j};
        V = V - lambda(i,j)*M_N{i,j};
        
    end
end
cons = cons + (v>=0);
cons = cons + (V>=0);

% dV must be negative semidefinite
for i = 1:N+3
    for j = 1:N+3
        dv = dv + eta(i,j)*m_Np1{i,j};
        dV = dV + eta(i,j)*M_Np1{i,j};
    end
end
cons = cons + (dv<=0);
cons = cons + (dV<=0);

% Break homogeneity
cons = cons + (trace(P)>=1);

obj = 0;

% Solver settings
solver_opt = sdpsettings('solver',solver,'verbose',0);

solverDetails = optimize(cons,obj,solver_opt);

if solverDetails.problem==1 || solverDetails.problem==4 %infeasible or "numerical problem" -> declare infeasible
    flag = 0;
else
    flag = 1;
end

end


function [v,V,dv,dV,m,M] = SetupSDP(mu,L,alpha,beta,gamma,rho,N,P,p,K)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 1) INITIALIZATION
x = cell(N+K+2,1);  % \bar{x}_k^(K) = x_N{N+k+1},   k=-N,...,K
y = cell(N+2,1);    % \bar{y}_k^(K) = y_N{k+1},     k= 0,...,K
g = cell(N+2,1);    % \bar{g}_k^(K) = g_N{k+1},     k= 0,...,K
f = cell(N+2,1);    % \bar{f}_k^(K) = f_N{k+1},     k= 0,...,K

% Initial condition
e_NpKp2 = eye(N+K+2);
for k = -N:0
    x{k+N+1} = e_NpKp2(k+N+1,:);
end

% Function and gradient values
e_Kp1 = eye(K+1);
for k = 0:K
    g{k+1} = e_NpKp2(k+N+2,:);
    f{k+1} = e_Kp1(k+1,:);
end

% Optimal point
y{K+2} = zeros(1,N+K+2);
g{K+2} = zeros(1,N+K+2);
f{K+2} = zeros(1,K+1);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 2) METHOD
for k = 0:K
    y{k+1}   = zeros(1,N+K+2);
    x{k+N+2} = -alpha*g{k+1};
    
    for j = 0:N
        y{k+1}   = y{k+1}  + gamma(j+1)*x{k-j+N+1};
        x{k+N+2} = x{k+N+2} + beta(j+1)*x{k-j+N+1};
    end
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 3) INTERPOLATION CONDITIONS
MM = [-mu*L, mu*L, mu, -L; mu*L, -mu*L, -mu, L; mu, -mu, -1, 1; -L, L, 1, -1];

m = cell(K+2);
M = cell(K+2);
for i = 1:K+2
    for j = 1:K+2
        m{i,j} = (L-mu)*(f{i}-f{j});
        M{i,j} = 1/2*[y{i}; y{j}; g{i}; g{j}]'*MM*[y{i}; y{j}; g{i}; g{j}];
        
        % Make sure M{i,j} is symmetric (should be, but cvx may complain)
        M{i,j} = (M{i,j} + M{i,j}')/2;
        
        % Make sure M{i,i} is exactly zero
        if i == j
            M{i,j} = zeros(size(M{i,j}));
        end
    end
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 4) LYAPUNOV FUNCTION
X_K = cell2mat(x(N+K+1:-1:K+1));
G_K = cell2mat(g(K+1:-1:K+1-N));
F_K = cell2mat(f(K+1:-1:K+1-N));

v = p'*F_K;
V = [X_K; G_K]'*P*[X_K; G_K];

% Calculate dV (if the basis dimension K is large enough)
if K > N
    X_Km1 = cell2mat(x(N+K:-1:K));
    G_Km1 = cell2mat(g(K:-1:K-N));
    F_Km1 = cell2mat(f(K:-1:K-N));

    vm1 = p'*F_Km1;
    Vm1 = [X_Km1; G_Km1]'*P*[X_Km1; G_Km1];

    dv = v - rho^2*vm1;
    dV = V - rho^2*Vm1;
else
    dv = [];
    dV = [];
end

end