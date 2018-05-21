#--------------------------------------------------------------
# MultilevelTT package
# example code for https://arxiv.org/abs/1802.09062
#--------------------------------------------------------------
# Code for 1D Poisson equation example,
# using AMEn solver via Python interface (see TTPy.jl)
module PoissonAMEn

using TT, TTFEM, TTAlg
using PyCall, TTPy

export fullSystem, amen1D, amen2D

# explicit assembly of preconditioned system
# (as required by amen solver)
function fullSystem(L::Integer, D::Integer)
  C = ttBPXdnC(L, D)

  f1 = ttrhsdn(L)
  f = f1
  for k = 2:D
      f = ttkron(f, f1)
  end
  g = matvec(C, f)
  svdtrunc!(g, svd!(g), 1e-14)

  Q = ttBPXdnQLaplace(L, D, 1)

  B = matmat(transpose(Q), Q)
  for k = 2:D
      Q = ttBPXdnQLaplace(L, D, k)
      B = add!(B, 1., matmat(transpose(Q), Q))
  end

  p = size(B[L],1)
  q = size(B[L],4)
  T = reshape(B[L],(p*2^D*2^D,q))
  S = reshape(B[L+1],(q,1))
  B[L] = reshape(T*S,(p,2^D,2^D,1))
  pop!(B)

  return B, g
end

function amen1D(L::Integer, ε::AbstractFloat)
    B, g = fullSystem(L, 1)
    C = ttBPXdnC(L, 1)
    u = amen(B, g, g, ε)
    Cu = matvec(C, u)
    return u, Cu
end

function amen2D(L::Integer, ε::AbstractFloat,
            kickrank::Integer = 4, nswp::Integer = 20)
    B, g = fullSystem(L, 2)
    u = amen(B, g, g, ε, kickrank, nswp)
    B = 0
    gc()

    C = ttBPXdnC(L, 2)
    Cu = matvec(C, u)

    return u, Cu
end

end
