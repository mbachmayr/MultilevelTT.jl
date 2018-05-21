#--------------------------------------------------------------
# MultilevelTT package
# example code for https://arxiv.org/abs/1802.09062
#--------------------------------------------------------------
# Code for 1D example with highly oscillating coefficient,
# using AMEn solver via Python interface (see TTPy.jl)
module MultiscaleAMEn

using TT, TTFEM, TTAlg, TTPy

export amenOscillatory1D

function amenOscillatory1D(L::Integer, ε::AbstractFloat, K::Number)
    Dinv = ttdiagm(add(2., ttones(L), 1., ttcos(L, K*π, .5)))
    Drhs = ttones(L)
    d, _ = stsolve(L, 1, x->add(1., matvec(Dinv,x), -1., Drhs), 3., 1., .5*ɛ, .75)
    D = ttdiagm(d)
    C = ttBPXdnC(L, 1)
    F = ttBPXdnQLaplace(L,1,1)
    p = size(F[L],1)
    q = size(F[L],4)
    T = reshape(F[L],(p*4,q))
    S = reshape(F[L+1],(q,1))
    F[L] = reshape(T*S,(p,2,2,1))
    pop!(F)
    Ft = transpose(F)
    f = ttrhsdn(L)
    g = matvec(C, f)
    svdtrunc!(g, svd!(g), .5*ɛ)
    B = matmat(Ft, matmat(D, F))
    u = amen(B, g, g, ε)
    Cu = matvec(C, u)
    Fu = matvec(F, u)
    return u, Cu, Fu
end

end
