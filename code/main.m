%% Lyapunov Functions for First-Order Methods: Tight Automated Convergence Guarantees

% This file produces the data for all figures in the paper:
% 
% A. Taylor, B. Van Scoy, L. Lessard, "Lyapunov Functions for First-Order
%  Methods: Tight Automated Convergence Guarantees," arXiv:####.####, 2018.
% 
% This code requires YALMIP (tested on version 20171121).

% SDP solver
solver = 'mosek';

% Folder where data is saved (must exist)
folder = '../data/';


%% Example

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Calculate the worst-case linear convergence rates of the following
% methods when applied to an L-smooth mu-strongly convex function:
%   1) Gradient method with step-size 1/L
%   2) Gradient method with step-size 2/(L+mu)
%   3) Heavy-ball method
%   4) Fast gradient method
%   5) Triple momentum method
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

mu = 1;
L  = 10;
FixedStepMethod(mu,L);


%% Figure 1: Worst-case linear convergence rate of fixed-step methods
clc;

% Number of grid points
N = 25;

% Bounds on rho for bisection
rho_bounds = [0,2];

% Tolerance for bisection
tol = 1e-4;

% Condition ratio
kappa = logspace(0,3,N);

GM  = zeros(1,N);
HBM = zeros(1,N);
FGM = zeros(1,N);
TMM = zeros(1,N);

fprintf('%3s\t%6s\t%6s\t%6s\t%6s\n','k','GM','HBM','FGM','TMM');

for k = 1:N
    mu = 1/kappa(k);
    L  = 1;
    
    % Gradient Method
    alpha = 1/L;
    GM(k) = FixedStepMethod(mu,L,alpha,1,1,rho_bounds,tol,solver);

    % Heavy Ball Method
    alpha  = 4/(sqrt(L)+sqrt(mu))^2;
    beta   = ((sqrt(L)-sqrt(mu))/(sqrt(L)+sqrt(mu)))^2;
    gamma  = 0;
    HBM(k) = FixedStepMethod(mu,L,alpha,[1+beta,-beta],[1+gamma,-gamma],rho_bounds,tol,solver);

    % Fast Gradient Method
    alpha  = 1/L;
    beta   = (sqrt(L)-sqrt(mu))/(sqrt(L)+sqrt(mu));
    gamma  = beta;
    FGM(k) = FixedStepMethod(mu,L,alpha,[1+beta,-beta],[1+gamma,-gamma],rho_bounds,tol,solver);

    % Triple Momentum Method
    RHO    = 1-sqrt(mu/L);
    alpha  = (1+RHO)/L;
    beta   = RHO^2/(2-RHO);
    gamma  = RHO^2/((1+RHO)*(2-RHO));
    TMM(k) = FixedStepMethod(mu,L,alpha,[1+beta,-beta],[1+gamma,-gamma],rho_bounds,tol,solver);
    
    fprintf('%3u\t%6.4f\t%6.4f\t%6.4f\t%6.4f\n',k,GM(k),HBM(k),FGM(k),TMM(k));
end

% Save data
save([folder 'data.mat'],'N','rho_bounds','tol','kappa','GM','HBM','FGM','TMM');


%% Plot Figure 1
clc; close all;

load([folder 'data.mat']);

figure;
semilogx(kappa,GM, 'b','linewidth',2); hold on;
semilogx(kappa,HBM,'r','linewidth',2);
semilogx(kappa,FGM,'m','linewidth',2);
semilogx(kappa,TMM,'g','linewidth',2);
ylim([0,1.1]);
xlabel('Condition ratio (\kappa)');
ylabel('Worst-case convergence rate (\rho)');
leg = legend('GM','HBM','FGM','TMM');
set(leg,'Location','Southeast');


%% Figure 2: Worst-case linear convergence rate of fixed-step methods with P>0 and p>0
clc;

% Number of grid points
N = 25;

% Bounds on rho for bisection
rho_bounds = [0,2];

% Tolerance for bisection
tol = 1e-4;

% Condition ratio
kappa = logspace(0,3,N);

FGM_PD = zeros(1,N);
TMM_PD = zeros(1,N);

fprintf('%3s\t%6s\t%6s\n','k','FGM','TMM');

for k = 1:N
    mu = 1/kappa(k);
    L  = 1;
    
    % Fast Gradient Method
    alpha = 1/L;
    beta  = (sqrt(L)-sqrt(mu))/(sqrt(L)+sqrt(mu));
    gamma = beta;
    FGM_PD(k) = FixedStepMethod(mu,L,alpha,[1+beta,-beta],[1+gamma,-gamma],rho_bounds,tol,solver,1);

    % Triple Momentum Method
    RHO   = 1-sqrt(mu/L);
    alpha = (1+RHO)/L;
    beta  = RHO^2/(2-RHO);
    gamma = RHO^2/((1+RHO)*(2-RHO));
    TMM_PD(k) = FixedStepMethod(mu,L,alpha,[1+beta,-beta],[1+gamma,-gamma],rho_bounds,tol,solver,1);
    
    fprintf('%3u\t%6.4f\t%6.4f\n',k,FGM_PD(k),TMM_PD(k));
end

% Save data
save([folder 'data_PositiveDefP.mat'],'N','rho_bounds','tol','kappa','FGM_PD','TMM_PD');


%% Plot Figure 2
clc; close all;

load([folder 'data.mat']);
load([folder 'data_PositiveDefP.mat']);

figure;
loglog(kappa,-1./(log(FGM)),    '-g','linewidth',2);  hold on;
loglog(kappa,-1./(log(FGM_PD)),'--g','linewidth',2);
loglog(kappa,-1./(log(TMM)),    '-m','linewidth',2);
loglog(kappa,-1./(log(TMM_PD)),'--m','linewidth',2);
ylim([1e-1,1e2]);
xlabel('Condition ratio (\kappa)');
ylabel('Evaluations to convergence (-1/log(\rho))');
leg = legend('FGM','FGM (pos. def.)','TMM','TMM (pos. def)');
set(leg,'Location','Southeast');


%% Figure 3: Worst-case linear convergence rate of methods with exact line-search
clc;

% Number of grid points
N = 25;

% Bounds on rho for bisection
rho_bounds = [0,2];

% Tolerance for bisection
tol = 1e-4;

% Condition ratio
kappa = logspace(0,3,N);

SD     = zeros(1,N);
HBM_SS = zeros(1,N);

fprintf('%3s\t%6s\t%6s\n','k','SD','HBM');

for k = 1:N
    mu = 1/kappa(k);
    L  = 1;
    
    % Steepest Descent
    SD(k) = SteepestDescent(mu,L,rho_bounds,tol,solver);

    % Heavy-ball with subspace searches
    HBM_SS(k) = HeavyBallSubspaceSearch(mu,L,rho_bounds,tol,solver);
    
    fprintf('%3u\t%6.4f\t%6.4f\n',k,SD(k),HBM_SS(k));
end

% Save data
save([folder 'data_SubspaceSearches.mat'],'N','rho_bounds','tol','kappa','SD','HBM_SS');


%% Plot Figure 3
clc; close all;

load([folder 'data_SubspaceSearches.mat']);
load([folder 'data.mat']);

figure;
semilogx(kappa,SD,     '-b','linewidth',2); hold on;
semilogx(kappa,HBM_SS, '-r','linewidth',2);
semilogx(kappa,GM,    '--b','linewidth',2);
semilogx(kappa,TMM,   '--g','linewidth',2);
ylim([0,1.1]);
xlabel('Condition ratio (\kappa)');
ylabel('Worst-case convergence rate (\rho)');
leg = legend('GM (linesearch)','HBM (subspace search)','GM (fixed-step of 1/L)','TMM (fixed-step)');
set(leg,'Location','Southeast');


%% Figure 4: Worst-case linear convergence rate of restarted FGM
% Part 1 (without the min_N rho(N))
clc;

% Number of grid points for kappa
Nb_kappa = 20;

% Bounds on rho for bisection
rho_bounds = [0,2];

% Tolerance for bisection
tol = 1e-4;

% Condition ratio
kappa = logspace(0.7,3,Nb_kappa);

% Restart after N iterations
N    = [1, 5, 10, 20];
Nb_N = length(N);

% Theoretical upper bound on rho
theor = exp(-1./(exp(1)*sqrt(8*kappa)));

FGM_restart = zeros(Nb_kappa,Nb_N);

count = 0;

fprintf('Total cases: %3u\n\n',Nb_N*Nb_kappa);

fprintf('%3s\t%3s\t%8s\t%6s\n','#','N','kappa','rho');

for i = 1:Nb_kappa
    for j = 1:Nb_N
        
        count = count + 1;
        
        mu = 1/kappa(i);
        L  = 1;
        
        % Tolerance for bisection
%         if kappa(i) <= 10  % N(j) >= 10 || 
%             tol = 1e-8;
%         else
%             tol = 1e-4;
%         end
        
        FGM_restart(i,j) = RestartedFGM(mu,L,N(j),rho_bounds,tol,solver);
        
        fprintf('%3u\t%3u\t%8.4f\t%6.4f\n',count,N(j),kappa(i),FGM_restart(i,j));
    end
end

% Save data
save([folder 'data_restarted_FGM.mat'],'kappa','N','FGM_restart','rho_bounds','Nb_N','Nb_kappa','theor');


%% Figure 4: Worst-case linear convergence rate of restarted FGM
% Part 2 (computation of min_N rho(N))
%
% Note: This section is computationally intensive. However, data is saved
% after each iteration, so the script may be stopped at any point with all
% previously computed data available.
clc;

% Number of grid points for kappa
Nb_kappa = 20;

% Bounds on rho for bisection
rho_bounds = [0,1];

% Tolerance for bisection
tol = 1e-4;

% Condition ratio
kappa = logspace(0.7,3,Nb_kappa);

% Maximum number of iterations before restarting
N_max = 35;

Optimal_N   = zeros(1,1);
Optimal_rho = zeros(1,1);

N_cur = 1;

fprintf('%3s  %6s  %9s  %11s  %13s  %15s\n','#','kappa','current N','current rho','prospective N','prospective rho');

for i = 1:Nb_kappa
    
    mu = 1/kappa(i);
    L  = 1;
    
    % Calculate rho at the current value of N
    rho_cur = RestartedFGM(mu,L,N_cur,rho_bounds,tol,solver);
    
    % Now check if using a larger N results in a smaller rho
    for N_prospect = N_cur+1:N_max
        
        rho_prospect = RestartedFGM(mu,L,N_prospect,rho_bounds,tol,solver);
        
        fprintf('%3u  %6.2f  %9u  %11.4f  %13u  %15.4f\n',i,kappa(i),N_cur,rho_cur,N_prospect,rho_prospect);
        
        % If using N_prospect results in a smaller rho, then save that
        % that as the current value; otherwise, we have found the
        % optimal N for the current kappa.
        if rho_prospect < rho_cur
            N_cur   = N_prospect;
            rho_cur = rho_prospect;
        else
            break;
        end
    end
    
    % Save optimal values
    Optimal_rho(i) = rho_cur;
    Optimal_N(i)   = N_cur;
    
    % Save data to .mat file
    save([folder 'data_restarted_FGM_Opt_N.mat'],'kappa','N_max','Optimal_N','Optimal_rho','rho_bounds','Nb_kappa');

    % Stop if N is too large (computation takes too long)
    if N_cur >= N_max
        fprintf('STOP: reached max value for N\n');
        break
    end
end


%% Plot Figure 4
clc; close all;

figure;

% Part 1
load([folder 'data_restarted_FGM.mat']);

loglog((ones(Nb_N,1)*kappa).', -1./log(FGM_restart),'s-','linewidth',2);  hold on;
loglog(kappa.',-1./(log(theor)),'--k','linewidth',2); 

% Part 2
load([folder 'data_restarted_FGM_Opt_N.mat']);

loglog(kappa(1:length(Optimal_N)),-1./(log(Optimal_rho)),'--b','linewidth',2);

xlabel('Condition ratio (\kappa)');
ylabel('Evaluations to convergence (-1/log(\rho))');
leg = legend('N=1','N=5','N=10','N=20','theoretical','optimal');
set(leg,'Location','Southeast');
