#--------------------------------------------------------------
# MultilevelTT package
# example code for https://arxiv.org/abs/1802.09062
#--------------------------------------------------------------
# Interface to AMEn implementation in ttpy (Python TT-Toolbox),
# requires additional Python packages to be installed
module TTPy

# To install the required python packages via julia:
#
# using Conda, PyCall
# Conda.add("numpy")
# Conda.add("scipy")
# Conda.add("cython")
# @pyimport pip
# args = String[]
# push!(args, "install")
# push!(args, "--user")
# push!(args, "ttpy")
# pip.main(args)

using TT, PyCall

import TT.Tensor, TT.TensorMatrix

export pyvec, pymat, Tensor, TensorMatrix, amen

@pyimport tt
@pyimport tt.amen as amen_pkg

pyvec(t::Tensor) = tt.vector[:from_list](t)
pymat(t::TensorMatrix) = tt.matrix[:from_list](t)

Tensor(x::PyObject) = Tensor(tt.vector[:to_list](x))
TensorMatrix(x::PyObject) = TensorMatrix(tt.matrix[:to_list](x))

amen(A::TensorMatrix, b::Tensor, x::Tensor, ɛ::Float64,
    kickrank::Integer = 4, nswp::Integer = 20) =
  Tensor(amen_pkg.amen_solve(pymat(A), pyvec(b), pyvec(x), ɛ, kickrank, nswp))

end
