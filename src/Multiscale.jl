#--------------------------------------------------------------
# MultilevelTT package
# example code for https://arxiv.org/abs/1802.09062
#--------------------------------------------------------------
# Code for 1D example with highly oscillating coefficient,
# using soft thresholding solver
module Multiscale

using TT, TTFEM, TTAlg

export multiscaleRefSol, stsolveOscillatory1D

function ttExMultiscaleD(L::Int64, δ::Float64)
	n = [1; 2*ones(Int64,L); 1]

	r = [1; 6*ones(Int64,L+1); 1]
	V = [ zeros(r[l],n[l],r[l+1]) for l ∈ 1:L+2 ]

	V[1][1,1,1] = 1/4
	V[1][1,1,2] = -1/4
	V[1][1,1,3] = -sin(2*π/δ)/8
	V[1][1,1,4] = cos(2*π/δ)/8
	V[1][1,1,5] = sin(2*π/δ)/8
	V[1][1,1,6] = -cos(2*π/δ)/8

	for l ∈ 1:L

		V[l+1][1,1,1] = 1
		V[l+1][1,2,1] = 1

		V[l+1][2,1,1] = -1/2
		V[l+1][2,1,2] = 1/2
		V[l+1][2,2,1] = 1/2
		V[l+1][2,2,2] = 1/2

		V[l+1][3,1,3] = cos(π/δ/2)
		V[l+1][3,1,4] = -sin(π/δ/2)
		V[l+1][4,1,3] = sin(π/δ/2)
		V[l+1][4,1,4] = cos(π/δ/2)

		V[l+1][3,1,3] = cos(π/δ/2)
		V[l+1][3,1,4] = -sin(π/δ/2)
		V[l+1][4,1,3] = sin(π/δ/2)
		V[l+1][4,1,4] = cos(π/δ/2)

		V[l+1][3,2,3] = cos(π/δ/2)
		V[l+1][3,2,4] = sin(π/δ/2)
		V[l+1][4,2,3] = -sin(π/δ/2)
		V[l+1][4,2,4] = cos(π/δ/2)

		V[l+1][5,1,3] = -cos(π/δ/2)/2
		V[l+1][5,1,4] = sin(π/δ/2)/2
		V[l+1][5,1,5] = cos(π/δ/2)/2
		V[l+1][5,1,6] = -sin(π/δ/2)/2
		V[l+1][6,1,3] = -sin(π/δ/2)/2
		V[l+1][6,1,4] = -cos(π/δ/2)/2
		V[l+1][6,1,5] = sin(π/δ/2)/2
		V[l+1][6,1,6] = cos(π/δ/2)/2

		V[l+1][5,2,3] = cos(π/δ/2)/2
		V[l+1][5,2,4] = sin(π/δ/2)/2
		V[l+1][5,2,5] = cos(π/δ/2)/2
		V[l+1][5,2,6] = sin(π/δ/2)/2
		V[l+1][6,2,3] = -sin(π/δ/2)/2
		V[l+1][6,2,4] = cos(π/δ/2)/2
		V[l+1][6,2,5] = -sin(π/δ/2)/2
		V[l+1][6,2,6] = cos(π/δ/2)/2

		δ = 2*δ;
	end

	V[L+2][1,1,1] = 1
	V[L+2][2,1,1] = 0
	V[L+2][3,1,1] = 0
	V[L+2][4,1,1] = 1
	V[L+2][5,1,1] = 0
	V[L+2][6,1,1] = 0

	return V
end


function ttExMultiscale(L::Int64, δ::Float64)
	n = [1; 2*ones(Int64,L); 1]

	r = [1; 7*ones(Int64,L+1); 1]
	U = [ zeros(r[l],n[l],r[l+1]) for l ∈ 1:L+2 ]

	U[1][1,1,1] = 1/6 + δ^2/(4*π)^2
	U[1][1,1,2] = 1/8
	U[1][1,1,3] = -1/24
	U[1][1,1,4] = (sin(π/δ)-δ*cos(π/δ)/π)/π/16
	U[1][1,1,5] = (cos(π/δ)-δ*sin(π/δ)/π )/π/16
	U[1][1,1,6] = -sin(π/δ)/π/16
	U[1][1,1,7] = -cos(π/δ)/π/16

	K = δ^(1/L)

	for l ∈ 1:L
		δ = 2*δ;

		U[l+1][1,1,1] = 1
		U[l+1][1,2,1] = 1

		U[l+1][2,1,1] = -1/2
		U[l+1][2,1,2] = 1/2
		U[l+1][2,2,1] = 1/2
		U[l+1][2,2,2] = 1/2

		U[l+1][3,1,2] = -3/4
		U[l+1][3,1,3] = 1/4
		U[l+1][3,2,2] = 3/4
		U[l+1][3,2,3] = 1/4

		U[l+1][4,1,4] = cos(π/δ)*K
		U[l+1][4,1,5] = sin(π/δ)*K
		U[l+1][5,1,4] = -sin(π/δ)*K
		U[l+1][5,1,5] = cos(π/δ)*K

		U[l+1][4,2,4] = cos(π/δ)*K
		U[l+1][4,2,5] = -sin(π/δ)*K
		U[l+1][5,2,4] = sin(π/δ)*K
		U[l+1][5,2,5] = cos(π/δ)*K

		U[l+1][6,1,4] = -cos(π/δ)/2*K
		U[l+1][6,1,5] = -sin(π/δ)/2*K
		U[l+1][6,1,6] = cos(π/δ)/2*K
		U[l+1][6,1,7] = sin(π/δ)/2*K
		U[l+1][7,1,4] = sin(π/δ)/2*K
		U[l+1][7,1,5] = -cos(π/δ)/2*K
		U[l+1][7,1,6] = -sin(π/δ)/2*K
		U[l+1][7,1,7] = cos(π/δ)/2*K

		U[l+1][6,2,4] = cos(π/δ)/2*K
		U[l+1][6,2,5] = -sin(π/δ)/2*K
		U[l+1][6,2,6] = cos(π/δ)/2*K
		U[l+1][6,2,7] = -sin(π/δ)/2*K
		U[l+1][7,2,4] = sin(π/δ)/2*K
		U[l+1][7,2,5] = cos(π/δ)/2*K
		U[l+1][7,2,6] = sin(π/δ)/2*K
		U[l+1][7,2,7] = cos(π/δ)/2*K

	end

	U[L+2][1,1,1] = 1
	U[L+2][2,1,1] = 1
	U[L+2][3,1,1] = 1
	U[L+2][4,1,1] = cos(π/δ)
	U[L+2][5,1,1] = sin(π/δ)
	U[L+2][6,1,1] = cos(π/δ)
	U[L+2][7,1,1] = sin(π/δ)

	return U
end

function multiscaleRefSol(L::Integer, K::Number)
    ttuex0 = ttExMultiscale(L, 2/K)
    dttuex0 = ttExMultiscaleD(L, 2/K)
    ttuex = Tensor(L)
    dttuex = Tensor(L)
    if L > 2
        ttuex[2:(L-1)] = ttuex0[3:L]
        dttuex[2:(L-1)] = dttuex0[3:L]
    end

    ttuex[1] = 2.0^(-L/2)*reshape(reshape(ttuex0[1], (1,7))*reshape(ttuex0[2], (7,2*7)), (1,2,7))
    ttuex[L] = reshape(reshape(ttuex0[L+1], (7*2,7))*reshape(ttuex0[L+2], (7,1)), (7,2,1))
    scale!(ttuex, 4.)

    dttuex[1] = 2.0^(-L/2)*reshape(reshape(dttuex0[1], (1,6))*reshape(dttuex0[2], (6,2*6)), (1,2,6))
    dttuex[L] = reshape(reshape(dttuex0[L+1], (6*2,6))*reshape(dttuex0[L+2], (6,1)), (6,2,1))
    scale!(dttuex, 4.)

    return ttuex, dttuex
end


# iteration with 1D bpx preconditioning,
# case with oscillatory coefficient
function stsolveOscillatory1D(L::Int64, ɛ::Float64, K::Number = 1)
  normest = 25.
  invnormest = .5 * 3. # taking into account min of coeff

  C = ttBPXdnC(L, 1)

  #F = ttBPXdnQ(L)
  F = ttBPXdnQLaplace(L, 1, 1)
  p = size(F[L],1)
  q = size(F[L],4)
  display(F)
  display((p,q))
  T = reshape(F[L],(p*4,q))
  S = reshape(F[L+1],(q,1))
  F[L] = reshape(T*S,(p,2,2,1))
  pop!(F)

  Ft = transpose(F)

  Dinv = ttdiagm(add(2., ttones(L), 1., ttcos(L, K*π, .5)))
  Drhs = ttones(L)
  d, _ = stsolve(L, 1, x->add(1., matvec(Dinv,x), -1., Drhs), 3., 1., .5*ɛ, .75)

  D = ttdiagm(d)

  println("operator ranks: ", ranks(F)," / ", ranks(D),
    " / ", ranks(Ft))

  f = ttrhsdn(L)
  g = matvec(C, f)
  svdtrunc!(g, svd!(g), .5*ɛ);
  println("rhs ranks ", ranks(g))

  Θ = 0.5

  function residual(u::Tensor, δ::Float64)
    gδ = deepcopy(g)
    svdtrunc!(gδ, svd!(gδ), .1*δ);
    t = matvec(F, u)
    maxrank = [maximum(ranks(t))]
    svdtrunc!(t, svd!(t), .4*δ / sqrt(normest));
    push!(maxrank, maximum(ranks(t)))
    s = matvec(D, t)
    svdtrunc!(s, svd!(s), .4*δ / sqrt(normest));
    push!(maxrank, maximum(ranks(s)))
    r = matvec(Ft, s)
    push!(maxrank, maximum(ranks(r)))
    add!(r, -1., g)
    push!(maxrank, maximum(ranks(r)))
    svdtrunc!(r, svd!(r), .1*δ);
    push!(maxrank, maximum(ranks(r)))
    return r, maxrank
  end

  u, cdata = inexstsolve(L, 1, residual, normest, invnormest, ɛ, Θ)

  return u, matvec(C, u), cdata
end


end
