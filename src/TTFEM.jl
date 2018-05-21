#--------------------------------------------------------------
# MultilevelTT package
# example code for https://arxiv.org/abs/1802.09062
#--------------------------------------------------------------
# Representations of FE matrices with and without preconditioning
module TTFEM

using TT

export ttsin, ttcos,
    ttrhsdn, ttlaplacedd, ttlaplacedn, ttmassdn,
    ttBPXdnQ, ttBPXdnLambdaId, ttBPXdnLambdaD2, ttBPXdnQLaplace, ttBPXdnC

# grid values of sine function
function ttsin(L::Int64, ω::Float64, s::Float64 = 0.0)::Tensor
  t = Tensor(2, L, 2)
  t[1][1,:,1] = [1., cos(ω*0.5)]
  t[1][1,:,2] = [0., sin(ω*0.5)]
  for i = 2:(L-1)
    x = 2.0^(-i) * ω
    t[i][1,:,1] = [1., cos(x)]
    t[i][1,:,2] = [0., sin(x)]
    t[i][2,:,1] = [0., -sin(x)]
    t[i][2,:,2] = [1., cos(x)]
  end
  x = 2.0^(-L) * ω
  t[L][1,:,1] = [sin(s*x), sin((1.0+s)*x)]
  t[L][2,:,1] = [cos(s*x), cos((1.0+s)*x)]
  return t
end

# grid values of cosine function
ttcos(L::Int64, ω::Float64, s::Float64 = 0.0)::Tensor = ttsin(L, ω, s + (.5*π)/ω*2.0^L)

# constant right-hand side
# (homogeneous Dirichlet-Neumann boundary conditions, with L^2 scaling)
function ttrhsdn(L::Int64)::Tensor
  f = ttones(L); d = TT.ttdelta(L, [2 for i = 1:L]);
  add!(f, -0.5, d);
  scale!(f, 1./2.0^(L/2.))
end

# 1D Dirichlet-Dirichlet Laplacian
function ttlaplacedd(L::Int64)::TensorMatrix
  T = TensorMatrix(2, L, 3)
  T[1][1,:,:,1] = eye(2)
  T[1][1,:,:,2] = [ 0. 0.; 1. 0.]
  T[1][1,:,:,3] = [ 0. 1.; 0. 0.]
  T[1] *= 4.0
  for i = 2:(L-1)
    fill!(T[i], 0.)
    T[i][1,:,:,1] = eye(2)
    T[i][1,:,:,2] = [ 0. 0.; 1. 0.]
    T[i][1,:,:,3] = [ 0. 1.; 0. 0.]
    T[i][2,:,:,2] = [ 0. 1.; 0. 0.]
    T[i][3,:,:,3] = [ 0. 0.; 1. 0.]
    T[i] *= 4.0
  end
  T[L][1,:,:,1] = [ 2. -1.; -1. 2.]
  T[L][2,:,:,1] = [ 0. -1.; 0. 0.]
  T[L][3,:,:,1] = [ 0. 0.; -1. 0.]
  T[L] *= 4.0 * (1.0 + 2.0^(-L))^2
  return T
end

# 1D Dirichlet-Neumann Laplacian
function ttlaplacedn(L::Int64)::TensorMatrix
  T = TensorMatrix(2, L, 4)
  T[1][1,:,:,1] = eye(2)
  T[1][1,:,:,2] = [ 0. 0.; 1. 0.]
  T[1][1,:,:,3] = [ 0. 1.; 0. 0.]
  T[1][1,:,:,4] = [ 0. 0.; 0. 1.]
  T[1] *= 4.0
  for i = 2:(L-1)
    fill!(T[i], 0.)
    T[i][1,:,:,1] = eye(2)
    T[i][1,:,:,2] = [ 0. 0.; 1. 0.]
    T[i][1,:,:,3] = [ 0. 1.; 0. 0.]
    T[i][2,:,:,2] = [ 0. 1.; 0. 0.]
    T[i][3,:,:,3] = [ 0. 0.; 1. 0.]
    T[i][4,:,:,4] = [ 0. 0.; 0. 1.]
    T[i] *= 4.0
  end
  T[L][1,:,:,1] = [ 2. -1.; -1. 2.]
  T[L][2,:,:,1] = [ 0. -1.; 0. 0.]
  T[L][3,:,:,1] = [ 0. 0.; -1. 0.]
  T[L][4,:,:,1] = [ 0. 0.; 0. -1.]
  T[L] *= 4.0
  return T
end

function ttmassdn(L::Int64)::TensorMatrix
  T = TensorMatrix(2, L, 4)
  T[1][1,:,:,1] = eye(2)
  T[1][1,:,:,2] = [ 0. 0.; 1. 0.]
  T[1][1,:,:,3] = [ 0. 1.; 0. 0.]
  T[1][1,:,:,4] = [ 0. 0.; 0. 1.]
  for i = 2:(L-1)
    fill!(T[i], 0.)
    T[i][1,:,:,1] = eye(2)
    T[i][1,:,:,2] = [ 0. 0.; 1. 0.]
    T[i][1,:,:,3] = [ 0. 1.; 0. 0.]
    T[i][2,:,:,2] = [ 0. 1.; 0. 0.]
    T[i][3,:,:,3] = [ 0. 0.; 1. 0.]
    T[i][4,:,:,4] = [ 0. 0.; 0. 1.]
  end
  T[L][1,:,:,1] = [ 2./3. 1./6.; 1./6. 2./3.]
  T[L][2,:,:,1] = [ 0. 1./6.; 0. 0.]
  T[L][3,:,:,1] = [ 0. 0.; 1./6. 0.]
  T[L][4,:,:,1] = [ 0. 0.; 0. -1./3.]
  return T
end

function ttBPXdnC(L::Int64, D::Int64)::TensorMatrix
	c = 2.0.^(-collect(0:L))

	I = eye(2,2)
	J = [0 1; 0 0]
	I1 = [1 0; 0 0]
	I2 = [0 0; 0 1]
	O = zeros(2,2)

	U = [ I J  J' I2 ;
	      O J' O   O ;
	      O O  J   O ;
	      O O  O  I1 ]
	W = [ 1 2 0 1 0 0 0 0;
	 	    2 4 0 2 1 2 0 1;
				1 0 2 1 0 0 0 0;
				2 0 4 2 1 0 2 1;
				1 2 0 1 2 4 0 2;
				0 0 0 0 1 2 0 1;
				1 0 2 1 2 0 4 2;
				0 0 0 0 1 0 2 1 ] / 4/2
	UD = 1
	WD = 1
	for d ∈ 1:D
		UD = kron(U, UD)
		WD = kron(W, WD)
	end
	sz = repmat([2, 4], 2*D, 1)
	prm = collect(1:D)
	prm = [2*prm; 2*prm-1; 2*D+2*prm-1; 2*D+2*prm]
	UD = reshape(UD, sz...)
	UD = permutedims(UD, prm)
	UD = reshape(UD, 4^D, 2^D, 2^D, 4^D)
	WD = reshape(WD, sz...)
	WD = permutedims(WD, prm)
	WD = reshape(WD, 4^D, 2^D, 2^D, 4^D)

	W = zeros(2*4^D, 2^D, 2^D, 2*4^D);
	W[1:4^D,:,:,1:4^D] = UD;
	W[1:4^D,:,:,4^D+1:2*4^D] = UD;
	W[4^D+1:2*4^D,:,:,4^D+1:2*4^D] = WD;

	Q = [ copy(W) for l ∈ 1:L ]
	for l ∈ 1:L
		Q[l][1:4^D,:,:,4^D+1:2*4^D] *= c[l+1];
	end
  Q[1] = Q[1][1:1,:,:,:] + c[1]*Q[1][4^D+1:4^D+1,:,:,:]
	Q[L] = Q[L][:,:,:,4^D+1:4^D+1]

  return Q
end

function ttBPXdnLambdaD2(L::Int64, D::Int64, K::Int64)::TensorMatrix
	W = eye(2^D, 2^D) / 2^D
	W = reshape(W, (1,2^D,2^D,1))

	S = [2] / 2
	M = [2 0; 0 2/3] / 2
	V = 1;
	for k=1:D
		if k == K
			V = kron(S,V);
		else
			V = kron(M,V);
		end
	end
	V = reshape(V,(1,2^(D-1),2^(D-1),1))
	Λ  = [ copy(W) for l ∈ 1:L+1 ]
	Λ[L+1] = V;

	return Λ
end

function ttBPXdnLambdaId(L::Int64, D::Int64)::TensorMatrix
	W = eye(2^D, 2^D) / 2^D
	W = reshape(W, (1,2^D,2^D,1))

	M = [2 0; 0 2/3] / 2
	V = 1;
	for k=1:D
		V = kron(M,V);
	end
	V = reshape(V, (1,2^D,2^D,1))
	Λ  = [ copy(W) for l ∈ 1:L+1 ]
	Λ[L+1] = V;

	return Λ
end

function ttBPXdnQ(L::Int64, D::Int64, K::Int64)::TensorMatrix
		if K ∉ 0:D
			throw(ArgumentError("K should be in 0:D"))
		end
		if L < 2
			throw(ArgumentError("L should be larger than one"))
		end

		if K ∈ 1:D
			c = ones(L+1)
		else
			c = 2.0.^(-collect(0:L))
		end

    U0 = [  1  0  0  1  0  0  0  0 ;
            0  1  0  0  1  0  0  1 ;
            0  0  0  0  0  0  0  0 ;
            0  0  1  0  0  0  0  0 ;
            0  0  0  0  0  1  0  0 ;
            0  0  0  0  0  0  0  0 ;
            0  0  0  0  0  0  1  0 ;
            0  0  0  0  0  0  0  0 ]
    U0 = U0 * sqrt(2) * 1 * 1

    U1 = [  1  0  0  1  0  0  0  0 ;
            0  1  0  0  1  0  0  1 ;
            0  0  0  0  0  0  0  0 ;
            0  0  1  0  0  0  0  0 ;
            0  0  0  0  0  1  0  0 ;
            0  0  0  0  0  0  0  0 ;
            0  0  0  0  0  0  1  0 ;
            0  0  0  0  0  0  0  0 ]
    U1 = U1 * sqrt(2) * 2 * 1 / 2

    Y0 = [  2  4  0  2  0  0  0  0 ;
            2  4  0  2  0  0  0  0 ;
            2  0  4  2  0  0  0  0 ;
            2  0  4  2  0  0  0  0 ;
           -1 -2  0 -1  1  2  0  1 ;
            1  2  0  1  1  2  0  1 ;
           -1  0 -2 -1  1  0  2  1 ;
            1  0  2  1  1  0  2  1 ] / 2 / 2
    Y0 = Y0 * sqrt(2) * 1 / sqrt(2) / sqrt(2)

    Y1 = [  1  2  0  1 ;
            1  2  0  1 ;
            1  0  2  1 ;
            1  0  2  1 ] / 2 / 2
    Y1 = Y1 * sqrt(2) * 2 / sqrt(2) / sqrt(2)

    UT0 = [  1  0  0  1  1  0  0  1 ;
             1  1  0  1 -1  1  0 -1 ;
             0  0  0  0  0  0  0  0 ;
             0  0  1  0  0  0  1  0 ;
             0  1  0  0  0 -1  0  0 ;
             0  0  0  0  0  0  0  0 ;
             0  0  1  0  0  0 -1  0 ;
             0  0  0  0  0  0  0  0 ]
    UT0 = UT0 * sqrt(2) * 1 * 1

    UT1 = [  1  0  0  1 ;
            -1  1  0 -1 ;
             0  0  0  0 ;
             0  0  1  0 ;
             0 -1  0  0 ;
             0  0  0  0 ;
             0  0 -1  0 ;
             0  0  0  0 ]
    UT1 = UT1 * sqrt(2) * 2 * 1 / 2

    TY0 = [  1  2  0  1  1  2  0  1 ;
             3  6  0  3  1  2  0  1 ;
             1  0  2  1  1  0  2  1 ;
             3  0  6  3  1  0  2  1 ;
             3  6  0  3 -1 -2  0 -1 ;
             1  2  0  1 -1 -2  0 -1 ;
             3  0  6  3 -1  0 -2 -1 ;
             1  0  2  1 -1  0 -2 -1 ] / 2 / 2
    TY0 = TY0 * sqrt(2) * 1 / sqrt(2) / sqrt(2)

    TY1 = [  1  2  0  1 ;
             1  2  0  1 ;
             1  0  2  1 ;
             1  0  2  1 ;
            -1 -2  0 -1 ;
            -1 -2  0 -1 ;
            -1  0 -2 -1 ;
            -1  0 -2 -1 ] / 2 / 2
    TY1 = TY1 * sqrt(2) * 2 / sqrt(2) / sqrt(2)

    M0 = [ 1 ;
           0 ;
           0 ;
           0 ;
           0 ;
           1 ;
           0 ;
           0 ] / 2

    M1 = [ 1 ;
           0 ] / 2
    M1 = M1 * 2

	U  = 1
	UT = 1
	TY = 1
	Y  = 1
	M  = 1
	for k ∈ 1:D
		if k == K
			U  = kron(U1, U)
			UT = kron(UT1, UT)
			TY = kron(TY1, TY)
			Y  = kron(Y1, Y)
			M  = kron(M1, M)
		else
			U  = kron(U0, U)
			UT = kron(UT0, UT)
			TY = kron(TY0, TY)
			Y  = kron(Y0, Y)
			M  = kron(M0, M)
		end
	end

	prm = collect(1:D)
	prm = [2*prm; 2*prm-1; 2*D+2*prm-1; 2*D+2*prm]

	sz = repmat([2, 4], 2*D, 1)
  U = reshape(U, sz...)
	U = permutedims(U, prm)
	sz = sz[collect(prm)]
	sz = prod(reshape(sz, (D, 4)), 1)
	U = reshape(U, sz...)

	sz = repmat([2, 4], 2*D, 1)
	if K ∈ 1:D
		sz[2*D+2*K] = 2
	end
  UT = reshape(UT, sz...)
	UT = permutedims(UT, prm)
	sz = sz[collect(prm)]
	sz = prod(reshape(sz, (D, 4)), 1)
	UT = reshape(UT, sz...)

	sz = repmat([2, 4], 2*D, 1)
	if K ∈ 1:D
		sz[2*D+2*K] = 2
	end
  TY = reshape(TY, sz...)
	TY = permutedims(TY, prm)
	sz = sz[collect(prm)]
	sz = prod(reshape(sz, (D, 4)), 1)
	TY = reshape(TY, sz...)

	sz = repmat([2, 4], 2*D, 1)
	if K ∈ 1:D
		sz[2*K] = 2
		sz[2*D+2*K] = 2
	end
  Y = reshape(Y, sz...)
	Y = permutedims(Y, prm)
	sz = sz[collect(prm)]
	sz = prod(reshape(sz, (D, 4)), 1)
	Y = reshape(Y, sz...)

	sz = repmat([2, 4], D, 1)
	sz = [sz; ones(Int64, 2*D, 1)]
	if K ∈ 1:D
		sz[2*K-1] = 1
		sz[2*K] = 2
	end
  M = reshape(M, sz...)
	M = permutedims(M, prm)
	sz = sz[collect(prm)]
	sz = prod(reshape(sz, (D, 4)), 1)
	M = reshape(M, sz...)

	Q = TensorMatrix(L+1)
	Q[1] = cat(4, U[1:1,:,:,:], c[2]*UT[1:1,:,:,:]+c[1]*TY[1:1,:,:,:]);
	O = zeros(size(Y, 1), 2^D, 2^D, size(U, 4))
	for ℓ ∈ 2:L-1
		Q[ℓ]= cat(4, cat(1, U, O), cat(1, c[ℓ+1]*UT, Y))
	end
	Q[L] = cat(1, c[L+1]*UT, Y)
	Q[L+1] = M

  return Q
end




function ttBPXdnQLaplace(L::Int64, D::Int64, K::Int64)::TensorMatrix
		if K ∉ 0:D
			throw(ArgumentError("K should be in 0:D"))
		end
		if L < 2
			throw(ArgumentError("L should be larger than one"))
		end

		if K ∈ 1:D
			c = ones(L+1)
		else
			c = 2.0.^(-collect(0:L))
		end


    U0 = [  1  0  0  1  0  0  0  0 ;
            0  1  0  0  1  0  0  1 ;
            0  0  0  0  0  0  0  0 ;
            0  0  1  0  0  0  0  0 ;
            0  0  0  0  0  1  0  0 ;
            0  0  0  0  0  0  0  0 ;
            0  0  0  0  0  0  1  0 ;
            0  0  0  0  0  0  0  0 ]
    U0 = U0 * sqrt(2) * 1 * 1 / sqrt(2)

    U1 = [  1  0  0  1  0  0  0  0 ;
            0  1  0  0  1  0  0  1 ;
            0  0  0  0  0  0  0  0 ;
            0  0  1  0  0  0  0  0 ;
            0  0  0  0  0  1  0  0 ;
            0  0  0  0  0  0  0  0 ;
            0  0  0  0  0  0  1  0 ;
            0  0  0  0  0  0  0  0 ]
    U1 = U1 * sqrt(2) * 2 * 1 / 2 / sqrt(2)

    Y0 = [  2  4  0  2  0  0  0  0 ;
            2  4  0  2  0  0  0  0 ;
            2  0  4  2  0  0  0  0 ;
            2  0  4  2  0  0  0  0 ;
           -1 -2  0 -1  1  2  0  1 ;
            1  2  0  1  1  2  0  1 ;
           -1  0 -2 -1  1  0  2  1 ;
            1  0  2  1  1  0  2  1 ] / 2 / 2
    Y0 = Y0 * sqrt(2) * 1 / sqrt(2) / sqrt(2) / sqrt(2)

    Y1 = [  1  2  0  1 ;
            1  2  0  1 ;
            1  0  2  1 ;
            1  0  2  1 ] / 2 / 2
    Y1 = Y1 * sqrt(2) * 2 / sqrt(2) / sqrt(2) / sqrt(2)

    UT0 = [  1  0  0  1  1  0  0  1 ;
             1  1  0  1 -1  1  0 -1 ;
             0  0  0  0  0  0  0  0 ;
             0  0  1  0  0  0  1  0 ;
             0  1  0  0  0 -1  0  0 ;
             0  0  0  0  0  0  0  0 ;
             0  0  1  0  0  0 -1  0 ;
             0  0  0  0  0  0  0  0 ]
    UT0 = UT0 * sqrt(2) * 1 * 1 / sqrt(2)

    UT1 = [  1  0  0  1 ;
            -1  1  0 -1 ;
             0  0  0  0 ;
             0  0  1  0 ;
             0 -1  0  0 ;
             0  0  0  0 ;
             0  0 -1  0 ;
             0  0  0  0 ]
    UT1 = UT1 * sqrt(2) * 2 * 1 / 2 / sqrt(2)

    TY0 = [  1  2  0  1  1  2  0  1 ;
             3  6  0  3  1  2  0  1 ;
             1  0  2  1  1  0  2  1 ;
             3  0  6  3  1  0  2  1 ;
             3  6  0  3 -1 -2  0 -1 ;
             1  2  0  1 -1 -2  0 -1 ;
             3  0  6  3 -1  0 -2 -1 ;
             1  0  2  1 -1  0 -2 -1 ] / 2 / 2
    TY0 = TY0 * sqrt(2) * 1 / sqrt(2) / sqrt(2) / sqrt(2)

    TY1 = [  1  2  0  1 ;
             1  2  0  1 ;
             1  0  2  1 ;
             1  0  2  1 ;
            -1 -2  0 -1 ;
            -1 -2  0 -1 ;
            -1  0 -2 -1 ;
            -1  0 -2 -1 ] / 2 / 2
    TY1 = TY1 * sqrt(2) * 2 / sqrt(2) / sqrt(2) / sqrt(2)

    M0 = [ 1         ;
           0         ;
           0         ;
           0         ;
           0         ;
           1/sqrt(3) ;
           0         ;
           0         ] / 2

    M1 = [ 1 ;
           0 ] / 2
    M1 = M1 * 2

	U  = 1
	UT = 1
	TY = 1
	Y  = 1
	M  = 1
	for k ∈ 1:D
		if k == K
			U  = kron(U1, U)
			UT = kron(UT1, UT)
			TY = kron(TY1, TY)
			Y  = kron(Y1, Y)
			M  = kron(M1, M)
		else
			U  = kron(U0, U)
			UT = kron(UT0, UT)
			TY = kron(TY0, TY)
			Y  = kron(Y0, Y)
			M  = kron(M0, M)
		end
	end

	prm = collect(1:D)
	prm = [2*prm; 2*prm-1; 2*D+2*prm-1; 2*D+2*prm]

	sz = repmat([2, 4], 2*D, 1)
  U = reshape(U, sz...)
	U = permutedims(U, prm)
	sz = sz[collect(prm)]
	sz = prod(reshape(sz, (D, 4)), 1)
	U = reshape(U, sz...)

	sz = repmat([2, 4], 2*D, 1)
	if K ∈ 1:D
		sz[2*D+2*K] = 2
	end
  UT = reshape(UT, sz...)
	UT = permutedims(UT, prm)
	sz = sz[collect(prm)]
	sz = prod(reshape(sz, (D, 4)), 1)
	UT = reshape(UT, sz...)

	sz = repmat([2, 4], 2*D, 1)
	if K ∈ 1:D
		sz[2*D+2*K] = 2
	end
  TY = reshape(TY, sz...)
	TY = permutedims(TY, prm)
	sz = sz[collect(prm)]
	sz = prod(reshape(sz, (D, 4)), 1)
	TY = reshape(TY, sz...)

	sz = repmat([2, 4], 2*D, 1)
	if K ∈ 1:D
		sz[2*K] = 2
		sz[2*D+2*K] = 2
	end
  Y = reshape(Y, sz...)
	Y = permutedims(Y, prm)
	sz = sz[collect(prm)]
	sz = prod(reshape(sz, (D, 4)), 1)
	Y = reshape(Y, sz...)

	sz = repmat([2, 4], D, 1)
	sz = [sz; ones(Int64, 2*D, 1)]
	if K ∈ 1:D
		sz[2*K-1] = 1
		sz[2*K] = 2
	end
  M = reshape(M, sz...)
	M = permutedims(M, prm)
	sz = sz[collect(prm)]
	sz = prod(reshape(sz, (D, 4)), 1)
	M = reshape(M, sz...)

	Q = TensorMatrix(L+1)
	Q[1] = cat(4, U[1:1,:,:,:], c[2]*UT[1:1,:,:,:]+c[1]*TY[1:1,:,:,:]);
	O = zeros(size(Y, 1), 2^D, 2^D, size(U, 4))
	for ℓ ∈ 2:L-1
		Q[ℓ]= cat(4, cat(1, U, O), cat(1, c[ℓ+1]*UT, Y))
	end
	Q[L] = cat(1, c[L+1]*UT, Y)
	Q[L+1] = M

  return Q
end


end
