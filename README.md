# Lyapunov Functions for First-Order Methods: Tight Automated Convergence Guarantees

This code can be used to analyze the worst-case linear convergence rate of first-order optimization algorithms applied to smooth strongly convex functions as described in the paper:

> [1] A. Taylor, B. Van Scoy, L. Lessard, "Lyapunov Functions for First-Order Methods: Tight Automated Convergence Guarantees," arXiv:####.####, 2018.

## Getting Started

To run the code, simply download the [code](/code) folder and add it to the path in Matlab.

For convenience, the data files produced by `main.m` are contained in the [data](/data) folder. 

**Note:** This code requires [YALMIP](https://yalmip.github.io/) along with a suitable SDP solver (e.g., Sedumi, SDPT3, Mosek).


## List of files

- [`main.m`](main.m) generates all the figures in [1] and illustrates how to use each function below.
- [`FixedStepMethod.m`](FixedStepMethod.m) calculates the worst-case linear convergence rate of a fixed-step method applied to a smooth strongly convex function.
- [`SteepestDescent.m`](SteepestDescent.m) analyzes the steepest descent method using an exact line search.
- [`HeavyBallSubspaceSearch.m`](HeavyBallSubspaceSearch.m) analyzes the heavy-ball method using an exact 2-dimensional subspace search at each iteration.
- [`RestartedFGM.m`](FixedStepMethod.m) analyzes the standard fast gradient method in (Nesterov, 1983) where the iterations are restarted every N iterations.


## Example

The following code calculates the worst-case linear convergence rate of the following methods when applied to an L-smooth mu-strongly convex function:
 1) Gradient method with step-size 1/L
 2) Gradient method with step-size 2/(L+mu)
 3) Heavy-ball method
 4) Fast gradient method
 5) Triple momentum method

```Matlab
mu = 1;   % strong convexity parameter
L  = 10;  % Lipschitz parameter of gradient

FixedStepMethod(mu,L);

% Results printed to screen:
% 
%        Method     Rate
%      GM (1/L) : 0.9000
% GM (2/(L+mu)) : 0.8182
%           HBM : 0.8602
%           FGM : 0.7518
%           TMM : 0.6838
```

## Authors

- [**Adrien Taylor**](http://www.di.ens.fr/~ataylor/)
- [**Bryan Van Scoy**](vanscoy@wisc.edu)
- [**Laurent Lessard**](http://www.laurentlessard.com/)
