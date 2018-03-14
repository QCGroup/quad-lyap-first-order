function [rho,P,p,lambda,eta] = SteepestDescent(mu,L,rho,tol,solver)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Finds a quadratic Lyapunov function for steepest descent
% applied to a smooth strongly convex function.
% 
% Note: This function requires YALMIP
% 
% Inputs:
%   mu      -  strong convexity parameter
%   L       -  smoothness parameter (Lipschitz constant of gradient)
%   rho     -  either convergence rate to test, or upper and lower bounds for bisection
%   tol     -  tolerance for bisection
%
% Outputs:
%   P,p,lambda  -  Lyapunov function parameters
%   eta         -  slack term coefficients in decrease of Lyapunov function
%   rho         -  convergence rate
% 
% Usage:
%   []                   = SteepestDescent(mu,L)
%   [  ~,P,p,lambda,eta] = SteepestDescent(mu,L,rho,tol)
%   [rho,P,p,lambda,eta] = SteepestDescent(mu,L,[rho_lower,rho_upper],tol)
% 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% CHECK INPUTS
if nargin>=4 && length(rho)==1
    
    % If given rho, solve the SDP to find a Lyapunov function (if possible)
    [~,P,p,lambda,eta] = SolveSDP(mu,L,rho,solver);
    
else
    % Bounds to use for bisection
    %  - Use [rho_lower, rho_upper] if specified
    %  - Otherwise use [0,1]
    if nargin>=4 && length(rho)==2
        rho_lower = rho(1);
        rho_upper = rho(2);
    else
        rho_lower = 0;
        rho_upper = 1;
    end
    
    % Default tolerance for bisection
    if nargin<4 || isempty(tol)
        tol = 1e-4;
    end
    
    % Perform bisection on rho to find the minimum rho for which the SDP is
    % feasible
    while rho_upper-rho_lower >= tol
        
        rho = (rho_lower + rho_upper)/2;
        
        flag = SolveSDP(mu,L,rho,solver);
        
        % Update bounds
        if flag
            rho_upper = rho;
        else
            rho_lower = rho;
        end
    end
    
    % Return the solution at the minimum rho for which the problem is feasible
    rho = rho_upper;
    [~,P,p,lambda,eta] = SolveSDP(mu,L,rho,solver);
end

end


function [flag,P,p,lambda,eta] = SolveSDP(mu,L,rho,solver)

% SDP variables
P=sdpvar(2);
p=sdpvar(1,1);
eta=sdpvar(3,3,'full');
cons=(eta>=0);
lambda=sdpvar(2,2,'full');
nu=sdpvar(2,1);

[v,V, ~, ~,  m_N,  M_N, ~] = SetupSDP(mu,L,rho,P,p,0);
[~,~,dv,dV,m_Np1,M_Np1, A] = SetupSDP(mu,L,rho,P,p,1);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 5) SEMIDEFINITE PROGRAM

% V must be positive definite
for i = 1:2
    for j = 1:2
        v = v - lambda(i,j)*m_N{i,j};
        V = V - lambda(i,j)*M_N{i,j};
    end
end
cons=cons+(v>=0);
cons=cons+(V>=0);

% dV must be negative semidefinite
for i = 1:3
    for j = 1:3
        dv = dv + eta(i,j)*m_Np1{i,j};
        dV = dV + eta(i,j)*M_Np1{i,j};
    end
end
cons=cons+(dv<=0);
cons=cons+(dV+nu(1)*A{1}+nu(2)*A{2}<=0);

% Break homogeneity
cons=cons+(trace(P)>=1);

obj = 0;

% Solver settings
solver_opt = sdpsettings('solver',solver,'verbose',0);

solverDetails = optimize(cons,obj,solver_opt);

if solverDetails.problem==1 || solverDetails.problem==4 %infeas. or "numerical problem" -> declare infeas.
    flag=0;
else
    flag=1;
end

end


function [v,V,dv,dV,m,M,A] = SetupSDP(mu,L,rho,P,p,K) %K=0 or K=1

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 1) INITIALIZATION
x = cell(K+2,1);    
g = cell(K+2,1);    
f = cell(K+2,1);

% Initial condition
e_NpKp2 = eye(2*K+2);
for k = 0:K
    x{k+1} = e_NpKp2(1+k,:);
end

% Function and gradient values
e_Kp1 = eye(K+1);
for k = 0:K
    g{k+1} = e_NpKp2(2+K+k,:);
    f{k+1} = e_Kp1(k+1,:);
end
% Optimal point
x{K+2} = zeros(1,2*K+2);
g{K+2} = zeros(1,2*K+2);
f{K+2} = zeros(1,K+1);
y=x;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 2) METHOD
A=cell(2,1);
if K==1
    A{1}=[x{1}; x{2}; g{2}].'*[0 0 -1; 0 0 1; -1 1 0]*[x{1}; x{2}; g{2}];
    A{2}=[g{1}; g{2}].'*[0 1; 1 0]*[g{1}; g{2}];
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
X_K = cell2mat(x(K+1));
G_K = cell2mat(g(K+1));
F_K = cell2mat(f(K+1));

v = p'*F_K;
V = [X_K; G_K]'*P*[X_K; G_K];

% Calculate dV (if the basis dimension K is large enough)
if K==1
    X_Km1 = cell2mat(x(1));
    G_Km1 = cell2mat(g(1));
    F_Km1 = cell2mat(f(1));

    vm1 = p'*F_Km1;
    Vm1 = [X_Km1; G_Km1]'*P*[X_Km1; G_Km1];

    dv = v - rho^2*vm1;
    dV = V - rho^2*Vm1;
else
    dv = [];
    dV = [];
end

end