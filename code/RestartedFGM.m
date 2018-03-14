function [rho,P,p,lambda,eta] = RestartedFGM(mu,L,horizon,rho,tol,solver)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Finds a quadratic Lyapunov function for a restarted version of FGM
% (method for smooth convex minimization) applied to a smooth strongly
% convex function.
%
% Note: This function requires YALMIP
%
% Inputs:
%   mu      -  strong convexity parameter
%   L       -  smoothness parameter (Lipschitz constant of gradient)
%   horizon -  inner iterations of FGM
%   rho     -  either convergence rate to test, or upper and lower bounds for bisection
%   tol     -  tolerance for bisection
%
% Outputs:
%   P,p,lambda  -  Lyapunov function parameters
%   eta         -  slack term coefficients in decrease of Lyapunov function
%   rho         -  convergence rate
%
% Usage:
%   [  ~,P,p,lambda,eta] = RestartedFGM(mu,L,horizon,rho)
%   [rho,P,p,lambda,eta] = RestartedFGM(mu,L,horizon,[rho_lower,rho_upper],tol)
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% CHECK INPUTS
if nargin>=4 && length(rho)==1
    
    % If given rho, solve the SDP to find a Lyapunov function (if possible)
    [~,P,p,lambda,eta] = SolveSDP(mu,L,rho,horizon,solver);
    
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
        
        flag = SolveSDP(mu,L,rho,horizon,solver);
        
        % Update bounds
        if flag
            rho_upper = rho;
        else
            rho_lower = rho;
        end
    end
    
    % Return the solution at the minimum rho for which the problem is feasible
    rho = rho_upper;
    [~,P,p,lambda,eta] = SolveSDP(mu,L,rho,horizon,solver);
end
rho = rho.^(1/horizon/2);  % Better conditioning that way!
end


function [flag,P,p,lambda,eta] = SolveSDP(mu,L,rho,horizon,solver)

% SDP variables
P=sdpvar(2);
p=sdpvar(1,1);
eta=sdpvar(horizon+2,horizon+2,'full');
cons=(eta>=0);
lambda=sdpvar(2,2,'full');

[v,V, ~, ~,  m_N,  M_N] = SetupSDP(mu,L,rho,P,p,1,horizon);
[~,~,dv,dV,m_Np1,M_Np1] = SetupSDP(mu,L,rho,P,p,horizon+1,horizon);

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
for i = 1:horizon+2
    for j = 1:horizon+2
        dv = dv + eta(i,j)*m_Np1{i,j};
        dV = dV + eta(i,j)*M_Np1{i,j};
    end
end
cons=cons+(dv<=0);
cons=cons+(dV<=0);

% Break homogeneity
cons=cons+(p>=1); %We break homogeneity with function values in this case.

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


function [v,V,dv,dV,m,M] = SetupSDP(mu,L,rho,P,p,K,N) %K=1 or K=horizon+1

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 1) INITIALIZATION

if K==1
    
    x=cell(2,1);
    g=cell(2,1);
    f=cell(2,1);
    x{1}=[1 0]; x{2}=[0 0];
    y=x;
    g{1}=[0 1]; g{2}=[0 0];
    f{1}=1; f{2}=0;
    
elseif K==N+1
    
    x = cell(N+1,1);
    y = cell(N+2,1);
    g = cell(N+2,1);
    f = cell(N+2,1);
    
    % Initial condition
    e_NpKp2 = eye(N+2);
    x{1} = e_NpKp2(1,:);
    y{1} = x{1};
    % Function and gradient values
    e_Kp1 = eye(N+1);
    for k = 0:N
        g{k+1} = e_NpKp2(2+k,:);
        f{k+1} = e_Kp1(k+1,:);
    end
    
    % Optimal point
    y{N+2} = zeros(1,N+2);
    g{N+2} = zeros(1,N+2);
    f{N+2} = zeros(1,N+1);

end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 2) METHOD

if K==N+1
theta = zeros(N,1); theta(1) = 1;

for k=1:N
    theta(k+1) = (1+sqrt(4*theta(k)^2+1))/2;
    coef = (theta(k)-1)/theta(k+1);
    x{k+1} = y{k}-1/L * g{k};
    y{k+1} = x{k+1}+coef*(x{k+1}-x{k});
end
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 3) INTERPOLATION CONDITIONS
MM = [-mu*L, mu*L, mu, -L; mu*L, -mu*L, -mu, L; mu, -mu, -1, 1; -L, L, 1, -1];

m = cell(K+1);
M = cell(K+1);
for i = 1:K+1
    for j = 1:K+1
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
X_K = cell2mat(y(K));
G_K = cell2mat(g(K));
F_K = cell2mat(f(K));
    
v = p'*F_K;
V = [X_K; G_K]'*P*[X_K; G_K];

% Calculate dV (if the basis dimension K is large enough)
if K==N+1
    X_Km1 = cell2mat(y(1));
    G_Km1 = cell2mat(g(1));
    F_Km1 = cell2mat(f(1));
    
    vm1 = p'*F_Km1;
    Vm1 = [X_Km1; G_Km1]'*P*[X_Km1; G_Km1];
    
    dv = v - rho*vm1;
    dV = V - rho*Vm1;
else
    dv = [];
    dV = [];
end

end