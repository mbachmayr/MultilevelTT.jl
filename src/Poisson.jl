#--------------------------------------------------------------
# MultilevelTT package
# example code for https://arxiv.org/abs/1802.09062
#--------------------------------------------------------------
# Code for 1D Poisson equation example,
# using soft thresholding solver
module Poisson

using TT, TTFEM, TTAlg

export quadraticRefSol, stsolve1D, stsolve2D

# exact solution of 1D test problem
function quadraticRefSol(L::Int64)
  t = Tensor(2, L, 3)
  x = .5;
  t[1] = reshape([0. 0. 1.; x^2 x 1.] , (1,2,3));
  t[1] /= 2*sqrt(2.)
  for i = 2:(L-1)
    x = 2.0^(-i);
    t[i] = reshape([ 1. 0. 0.; 0. 1. 0.; 0. 0. 1.;  1. 0. 0.;  2.*x 1. 0.; x^2 x 1.], (3,2,3))
    t[i] /= sqrt(2.)
  end
  x = 2.0^(-L);
  x0 = x
  x1 = x0 + x
  t[L] = reshape([ -1.; 2.*(1. - x0); x0*(2.-x0); -1.; 2.*(1. - x1); x1*(2.-x1)], (3, 2, 1))
  t[L] /= sqrt(2.)
  return t
end

# soft thresholding iteration with 1D BPX preconditioning,
# reduced-rank operator with intermediate recompression
function stsolve1D(L::Int64, ɛ::Float64)
  normest = 25.
  invnormest = .5

  C = ttBPXdnC(L, 1)
  F = ttBPXdnQLaplace(L, 1, 1)
  Ft = transpose(F)

  println("operator ranks: ", ranks(F)," / ", ranks(Ft))

  f = ttrhsdn(L)
  g = matvec(C, f)
  svdtrunc!(g, svd!(g), .005*ɛ);
  println("rhs ranks ", ranks(g))

  Θ = 0.5

  function residual(u::Tensor, δ::Float64)
    gδ = deepcopy(g)
    svdtrunc!(gδ, svd!(gδ), .1*δ);
    t = matvec(F, u)
    maxrank = [maximum(ranks(t))]
    svdtrunc!(t, svd!(t), .8*δ / sqrt(normest));
    push!(maxrank, maximum(ranks(t)))
    r = matvec(Ft, t)
    push!(maxrank, maximum(ranks(r)))
    add!(r, -1., g)
    push!(maxrank, maximum(ranks(r)))
    svdtrunc!(r, svd!(r), .1*δ);
    push!(maxrank, maximum(ranks(r)))
    return r, maxrank
  end

  u, cdata = inexstsolve(L, 1, residual, normest, invnormest, .995*ɛ, Θ, .95)

  return u, matvec(C, u), cdata
end

# soft thresholding iteration with 2D BPX preconditioning
function stsolve2D(L::Int64, ɛ::Float64)
  normest = 25.
  invnormest = .4

  C = ttBPXdnC(L, 2)

  Q11 = ttBPXdnQLaplace(L, 2, 1)
  Q12 = ttBPXdnQLaplace(L, 2, 2)
  Q11t = transpose(Q11)
  Q12t = transpose(Q12)

  f = ttkron(ttrhsdn(L), ttrhsdn(L))
  g = matvec(C, f)
  svdtrunc!(g, svd!(g), .005*ɛ);
  println("rhs ranks ", ranks(g))

  Θ = 0.5

  function residual(u::Tensor, δ::Float64)
	maxrank = Int64[]
    gδ = deepcopy(g)
    svdtrunc!(gδ, svd!(gδ), δ/7);

    s = matvec(Q11, u)
    svdtrunc!(s, svd!(s), δ / 7 / sqrt(normest));
    push!(maxrank, maximum(ranks(s)))

    r1 = matvec(Q11t, s)
	svdtrunc!(r1, svd!(r1), δ / 7);
    push!(maxrank, maximum(ranks(r1)))

	s = matvec(Q12, u)
    svdtrunc!(s, svd!(s), δ / 7 / sqrt(normest));
	push!(maxrank, maximum(ranks(s)))

	r2 = matvec(Q12t, s)
	svdtrunc!(r2, svd!(r2), δ / 7);
	push!(maxrank, maximum(ranks(r2)))

	add!(r1, r2)
	svdtrunc!(r1, svd!(r1), δ / 7);
	push!(maxrank, maximum(ranks(r1)))

    add!(r1, -1., g)
    svdtrunc!(r1, svd!(r1), δ / 7);
    push!(maxrank, maximum(ranks(r1)))
    return r1, maxrank
  end

  u, cdata = inexstsolve(L, 2, residual, normest, invnormest, .995*ɛ, Θ)

  return u, matvec(C, u), cdata
end

end
