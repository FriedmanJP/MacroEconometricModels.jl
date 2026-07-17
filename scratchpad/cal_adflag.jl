using MacroEconometricModels, Random
const A = MacroEconometricModels
for shift in (0.0, 4.0, 8.0, 15.0, 30.0)
    rng = MersenneTwister(2079); n=120; max_p=8
    e=randn(rng,n); y=zeros(n)
    for t in 3:n; y[t]=0.6*y[t-1]+0.3*y[t-2]+e[t]; end
    y[1:15] .+= shift
    dy=diff(y); Yfix=dy[(max_p+1):end]; Nfix=n-1-max_p
    fic(p,crit)=begin
        Xf=A._build_adf_matrix(y,dy,p,:constant); X=Xf[(max_p-p+1):end,:]; k=size(X,2)
        B=X\Yfix; r=Yfix-X*B; s2=sum(abs2,r)/(Nfix-k); ll=-Nfix/2*(log(2π)+log(s2)+1)
        crit==:aic ? -2ll+2k : -2ll+k*log(Nfix)
    end
    vic(p,crit)=begin
        Xf=A._build_adf_matrix(y,dy,p,:constant); Yv=dy[(p+1):end]; m=length(Yv); k=size(Xf,2)
        B=Xf\Yv; r=Yv-Xf*B; s2=sum(abs2,r)/(m-k); ll=-m/2*(log(2π)+log(s2)+1)
        crit==:aic ? -2ll+2k : -2ll+k*log(m)
    end
    ea=argmin([fic(p,:aic) for p in 0:max_p])-1; eb=argmin([fic(p,:bic) for p in 0:max_p])-1
    va=argmin([vic(p,:aic) for p in 0:max_p])-1
    sel_a=A.adf_select_lags(y,max_p,:constant,:aic); sel_b=A.adf_select_lags(y,max_p,:constant,:bic)
    println("shift=$shift: exp_aic=$ea sel_aic=$sel_a | exp_bic=$eb sel_bic=$sel_b | var_aic=$va | separate? ", ea!=va)
end
