function [rho,P,p,lambda,eta] = HeavyBallSubspaceSearch(mu,L,rho,tol,solver)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Finds a quadratic Lyapunov function for a subspace-search
% variant of the Heavy-ball method applied to a smooth strongly
% convex function.
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
%   []                   = HeavyBallSubspaceSearch(mu,L)
%   [  ~,P,p,lambda,eta] = HeavyBallSubspaceSearch(mu,L,rho,tol)
%   [rho,P,p,lambda,eta] = HeavyBallSubspaceSearch(mu,L,[rho_lower,rho_upper],tol)
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
P=sdpvar(4);
p=sdpvar(2,1);
eta=sdpvar(4,4,'full');
cons=(eta>=0);
lambda=sdpvar(3,3,'full');
nu=sdpvar(6,1);

[v,V, ~, ~,  m_N,  M_N, ~] = SetupSDP(mu,L,rho,P,p,1);
[~,~,dv,dV,m_Np1,M_Np1, A] = SetupSDP(mu,L,rho,P,p,2);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 5) SEMIDEFINITE PROGRAM

% V must be positive definite
for i = 1:3
    for j = 1:3
        v = v - lambda(i,j)*m_N{i,j};
        V = V - lambda(i,j)*M_N{i,j};
    end
end
cons=cons+(v>=0);
cons=cons+(V>=0);

% dV must be negative semidefinite
for i = 1:4
    for j = 1:4
        dv = dv + eta(i,j)*m_Np1{i,j};
        dV = dV + eta(i,j)*M_Np1{i,j};
    end
end
cons=cons+(dv<=0);
for i=1:6
    dV=dV+nu(i)*A{i};
end
cons=cons+(dV<=0);

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


function [v,V,dv,dV,m,M,A] = SetupSDP(mu,L,rho,P,p,K) %K=1 or K=2

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 1) INITIALIZATION

if K==1
    x = cell(3,1);
    g = cell(3,1);
    f = cell(3,1);
    
    % Initial condition
    e_NpKp2 = eye(4);
    for k = 0:1
        x{k+1} = e_NpKp2(1+k,:);
    end
    
    % Function and gradient values
    e_Kp1 = eye(2);
    for k = 0:1
        g{k+1} = e_NpKp2(3+k,:);
        f{k+1} = e_Kp1(k+1,:);
    end
    % Optimal point
    x{3} = zeros(1,4);
    g{3} = zeros(1,4);
    f{3} = zeros(1,2);
    y=x;
elseif K==2
    x = cell(5,1);
    g = cell(4,1);
    f = cell(4,1);
    
    % Initial condition
    e_NpKp2 = eye(7);
    for k = 0:3
        x{k+1} = e_NpKp2(1+k,:);
    end
    
    % Function and gradient values
    e_Kp1 = eye(3);
    for k = 0:2
        g{k+1} = e_NpKp2(5+k,:);
        f{k+1} = e_Kp1(k+1,:);
    end
    % Optimal point
    x{5} = zeros(1,7);
    g{4} = zeros(1,7);
    f{4} = zeros(1,3);
    y=cell(4,1);
    for k=1:4
        y{k}=x{1+k};
    end
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 2) METHOD
A=cell(6,1);
if K==2
    A{1}=[x{2}; x{3}; g{2}].'*[0 0 -1; 0 0 1; -1 1 0]*[x{2}; x{3}; g{2}];
    A{2}=[x{3}; x{4}; g{3}].'*[0 0 -1; 0 0 1; -1 1 0]*[x{3}; x{4}; g{3}];
    A{3}=[x{1}; x{2}; g{2}].'*[0 0 -1; 0 0 1; -1 1 0]*[x{1}; x{2}; g{2}];
    A{4}=[x{2}; x{3}; g{3}].'*[0 0 -1; 0 0 1; -1 1 0]*[x{2}; x{3}; g{3}];
    A{5}=[g{1}; g{2}].'*[0 1; 1 0]*[g{1}; g{2}];
    A{6}=[g{2}; g{3}].'*[0 1; 1 0]*[g{2}; g{3}];
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
X_K = cell2mat(y(K+1:-1:K));
G_K = cell2mat(g(K+1:-1:K));
F_K = cell2mat(f(K+1:-1:K));

v = p'*F_K;
V = [X_K; G_K]'*P*[X_K; G_K];

% Calculate dV (if the basis dimension K is large enough)
if K==2
    X_Km1 = cell2mat(y(K:-1:K-1));
    G_Km1 = cell2mat(g(K:-1:K-1));
    F_Km1 = cell2mat(f(K:-1:K-1));
    
    vm1 = p'*F_Km1;
    Vm1 = [X_Km1; G_Km1]'*P*[X_Km1; G_Km1];
    
    dv = v - rho^2*vm1;
    dV = V - rho^2*Vm1;
else
    dv = [];
    dV = [];
end

end