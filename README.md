# MultilevelTT

Example [Julia](https://julialang.org/) code for <https://arxiv.org/abs/1802.09062>. 

Includes a basic implementation of the tensor train format, assembly of the preconditioned low-rank representations developed in the paper, generic iterative solvers (including a wrapper for AMEn as implemented in [ttpy](https://github.com/oseledets/ttpy)) and some routines for running tests.

To install the package, run <code>Pkg.clone("https://github.com/mbachmayr/MultilevelTT.jl.git")</code>

`using MultilevelTT` only modifies the module search path `LOAD_PATH` so that the following modules can be loaded separately via `using`:
- `TT`, basic tensor train functions.
- `TTFEM`, routines for assembling tensor representations from <https://arxiv.org/abs/1802.09062>.
- `TTAlg`, iterative solvers implemented in Julia (following [this paper](http://dx.doi.org/10.1007/s10208-016-9314-z)).
- `TTPy`, wrapper for AMEn in [ttpy](https://github.com/oseledets/ttpy); requires this python package to be installed, see instructions in the source file. Required only by the `PoissonAMEn` and `MultiscaleAMEn` modules.
- `TTCondition`, auxiliary routines for evaluating representation condition numbers (see §4 in <https://arxiv.org/abs/1802.09062>).
- `Poisson`, test routines for 1D and 2D Poisson problems, using `TTAlg`.
- `Multiscale`, test routines for a 1D problem with oscillatory coefficient, using `TTAlg`.
- `PoissonAMEn`, test routines for 1D and 2D Poisson problems, using `TTPy`.
- `MultiscaleAMEn`, test routines for a 1D problem with oscillatory coefficient, using `TTPy`.
