using Distributions
using Optim

# Simulation parameters
N = 101 # N time intervals, N+1 dates
 # number of Monte Carlo simulations
Rbar = 0.1 # mean-reversion level of residual demand
Rho = 0.9 # reversion speed of residual demand
Sigma = 0.2 # volatility of residual demand noise
Cmin = 0.0 # minimum charge
C0   = 0.00 # initial charge
Cmax = 1.0 # maximum charge
K = 2.0 # cost multiplier for diesel generator
Gamma = 2.0 # power of cost of diesel generator
Kappa = 0.2 # on/off switching cost of diesel generator
Qm = 10.0 # multiplicative penalty for negative imbalance
R0 = 0.1 # initial residual demand
Amin = 0.05 # alpha_s minimum
Amax=10

function l(a)
    return K*a^Gamma
end

function S(c,m,r,a)
    return r-a+min(max(a-r,0),Cmax-c) - min(max(r-a,0),c)
end

# Running penalization
function f(c,m,r,a)
    if m==1
        return l(a)+(a==0)*Kappa + Qm*max(-S(c,m,r,a),0)
    else
        return l(a)+(a>=Amin)*Kappa + Qm*max(-S(c,m,r,a),0)
    end
end



# Sample generation
Nr=51
readfile=readdlm("one_dim_1_1000/$(Nr)_1_nopti")
GridB=readfile[:,2]
pop!(GridB)
GridR=[]
for t=1:N
    push!(GridR,Rho^(t-1)*R0+Sigma*((1-Rho^(t-1))/(1-Rho))*GridB)
end

Nc=51
function GenerateSample_C(n)
    return [Cmin + i/Nc*(Cmax-Cmin) for i=0:50]
end

GridC=[GenerateSample_C(n) for n=1:N];

function F1(c,r,a)
    return c+min(max(a-r,0),Cmax-c)-min(max(r-a,0),c)
end

function projC2(n,c,r,a) #Return the projection of cnext on GridC
    cnext=F1(c,r,a)
    @assert cnext<=Cmax && cnext>=Cmin
    if cnext >= GridC[n+1][length(GridC[n+1])]
        return (GridC[n+1][length(GridC[n+1])],GridC[n+1][length(GridC[n+1])])
    elseif cnext <= GridC[n+1][1]
        return (GridC[n+1][1],GridC[n+1][1])
    else
        int = searchsortedfirst(GridC[n+1],cnext)
        return (GridC[n+1][int-1],GridC[n+1][int])
    end
end

function projR(n,r,eps) #Return the projection of wnext on GridW[n+1] # eps= W[n+1]-W[n]
    rnext=Rbar*(1-Rho)+Rho*r+Sigma*eps
    #println("pnext ", pnext)
    if rnext>= GridR[n+1][length(GridR[n+1])]
        return GridR[n+1][length(GridR[n+1])]
    elseif rnext <= GridR[n+1][1]
        #println(wnext)
        return GridR[n+1][1]
    else
        int = searchsortedfirst(GridR[n+1],rnext)
        if rnext-GridR[n+1][int-1] < GridR[n+1][int]- rnext
            return GridR[n+1][int-1]
        else
            return GridR[n+1][int]
        end
    end
end

Strat=[Dict() for t=0:N] # Strategy dictionary
ValueFunction=[Dict() for t=0:N] # Value Function dictionary at all time 0,...,N
V=Dict() # Value function
for r in GridR[N]
    for c in GridC[N]
        for m=0:1
            V[(c,m,r)]=0
        end
    end
end

function phi0(x) # return the cdf of the Normal(0,1) law
    return cdf(Normal(),x)
end

Nnorm=21# number of points for the quantization of w
GridN=readdlm("one_dim_1_1000/$(Nnorm)_1_nopti")[:,2]
pop!(GridN)

function expectation(n, c, m, r, V, a) # return an approximation of the conditional expectation using quantization
    mnext=(a!=0)*1
    cnext=F1(c,r,a)
    @assert cnext>=Cmin && cnext<=Cmax
    cm, cp=projC2(n,c,r,a)
    lambda= cp>cm ? (cnext-cm)/(cp-cm):1
    rnext=projR(n,r,GridN[1])
    res=phi0((GridN[1]+GridN[2])/2)*(lambda*V[(cp,mnext,rnext)]+(1-lambda)*V[(cm,mnext,rnext)])
    for i=2:Nnorm-1
        rnext=projR(n,r,GridN[i])
        res += (phi0((GridN[i]+GridN[i+1])/2)-phi0((GridN[i]+GridN[i-1])/2))*(lambda*V[(cp,mnext,rnext)]+(1-lambda)*V[(cm,mnext,rnext)])
    end
    rnext=projR(n,r,GridN[Nnorm])
    res+=(1-phi0((GridN[Nnorm]+GridN[Nnorm-1])/2))*(lambda*V[(cp,mnext,rnext)]+(1-lambda)*V[(cm,mnext,rnext)])
    return res+ f(c,m,r,a)
end

function optimalExpectation(n,c,m,r,V)
    function funct_temp(a)
        return expectation(n,c,m,r,V,a)
    end
    if r-c<=0
        exp_a0=expectation(n,c,m,r,V,0)
        res=optimize(funct_temp,Amin,Amax, Brent())
        (minimizer,minimum)=(Optim.minimizer(res),Optim.minimum(res))
        if minimum<exp_a0
            return (minimizer,minimum)
        else
            return (0,exp_a0)
        end
    else
        res=optimize(funct_temp,max(Amin,r-c),Amax, Brent())
        (minimizer,minimum)=(Optim.minimizer(res),Optim.minimum(res))
        return (minimizer,minimum)
    end
end

function backward(n,V)
    Vback=Dict()
    for r in GridR[n]
        for c in GridC[n]
            # println(Amax-r+c)
            for m=0:1
                temp=optimalExpectation(n,c,m,r,V)
                Vback[(c,m,r)]=temp[2]
                ValueFunction[n][(c,m,r)]=temp[2]
                Strat[n][(c,m,r)]=temp[1]
            end
        end
    end
    return Vback
end

Vback=V
for t=N-1:-1:1  # compute the value function using Bellman scheme
    println(t)
    Vback=backward(t,Vback)
end

for m=0:1
    for n=1:N-1
        XC=GridC[n]
        XR=GridR[n]
        Z=zeros((Nc,Nr))
        for r=1:Nr
            for c=1:Nc
                Z[r,c]=Strat[n][(XC[c],m,XR[r])]
            end
        end
        writedlm("Decisionsn$(n)m$(m)",Z)
        writedlm("Cn$(n)",XC)
        writedlm("Rn$(n)",XR)
    end
end


# Plots and tests of the estimates

function plotStrat(m,n)
    XC=GridC[n]
    XR=GridR[n]
    Z=zeros((Nc,Nr))
    for r=1:Nr
        for c=1:Nc
            Z[r,c]=Strat[n][(XC[c],m,XR[r])]
        end
    end
    plotly(size=(600,400))
    heatmap(XC,XR,(c,r) -> Strat[n][(c,m,r)],c=ColorGradient([:white,:yellow,:red]),xlabel="C",ylabel="R",title="Decisions at t$(n) with m$(m)")
    png("DecisionsN$(N)_t$(n)m$(m)")
end

n=N-3
m=1
XC=GridC[n]
XR=GridR[n]
Z=zeros((Nc,Nr))
for r=1:Nr
    for c=1:Nc
        Z[r,c]=Strat[n][(XC[c],m,XR[r])]
    end
end

function plotValue(n,m)
    XC=GridC[n]
    XR=GridR[n]
    Z=zeros((Nc,Nr))
    for r=1:Nr
        for c=1:Nc
            Z[r,c]=ValueFunction[n][(XC[c],m,XR[r])]
        end
    end
    fig=figure("Value Function at time $(n) and m=$(m)", figsize=(8,6))
    imshow(Z, origin="lower",extent=[XC[1],XC[Nc],XR[1],XR[Nr]], aspect="auto")
    ax=axes()
    ax[:spines]["top"][:set_visible](false)
    ax[:spines]["right"][:set_visible](false)
    PyPlot.set_cmap("hot")
    xlabel("C")
    ylabel("R")
    title("ValueFunctionN$(N)_t$(n)m$(m)")
    colorbar()
    savefig("ValueFunctionN$(N)_t$(n)m$(m).pdf")
end

dN=Normal(0,1)

lenGridc= length(GridC[1])

function projeteC(n,c) #Return the projection of Cnext on GridC[n]
    if c>= GridC[n][length(GridC[n])]
        return GridC[n][length(GridC[n])]
    elseif c <= GridC[n][1]
        return GridC[n][1]
    else
        int = searchsortedfirst(GridC[n],c)
        if abs(c-GridC[n][int]) < abs(c-GridC[n][int-1])
            return GridC[n][int]
        else
            return GridC[n][int-1]
        end
    end
end

lenGridr= length(GridR[1])

function projeteR(r,n)
    if r>= GridR[n][lenGridr]
        return GridR[n][lenGridr]
    elseif r <= GridR[n][1]
        return GridR[n][1]
    else
        int = searchsortedfirst(GridR[n],r)
        if abs(r-GridR[n][int]) < abs(r-GridR[n][int-1])
            return GridR[n][int]
        else
            return GridR[n][int-1]
        end
    end
end

function testgeneral(NbTirages)
    R=[R0 for i in 1:NbTirages]
    global historyQ=[[] for n=1:N]
    Cquant=[C0 for i in 1:NbTirages]
    Jquant=[0 for i in 1:NbTirages]
    mQuant=[0 for i in 1:NbTirages]
    aQuant=[Strat[1][(C0,0,R0)] for i in 1:NbTirages]
    abench=[0 for i in 1:NbTirages]
    for n in 1:N-1
        for ind=1:20
            push!(historyQ[n],[Cquant[ind],mQuant[ind],R[ind]])
        end
        Jquant+=[f(Cquant[i],mQuant[i],R[i],aQuant[i]) for i in 1:NbTirages]
        dW=rand(dN,NbTirages)
        R=(1-Rho)*Rbar+Rho*R+Sigma*dW
        Cquant=[F1(Cquant[i],R[i],aQuant[i]) for i in 1:NbTirages]
        mQuant=[(aQuant[i] !=0) for i in 1:NbTirages]
        if n<N-1
            aQuant=[Strat[n+1][(projeteC(n+1,Cquant[i]),mQuant[i],projeteR(R[i],n+1))] for i in 1:NbTirages]
        end
    end
    EJquant=mean(Jquant)
    pop!(historyQ)
    return EJquant #, EJbench
end

print("MC forward")
res=[testgeneral(10000) for i=1:20]
println("mean", mean(res))
println("std", std(res))

for kk=1:20
res=testgeneral(1)
X=[i for i=1:N-1]
XC=[historyQ[i][1][1] for i=1:N-1]
XM=[historyQ[i][1][2] for i=1:N-1]
XR=[historyQ[i][1][3] for i=1:N-1]
writedlm("resC$(kk)",XC)
writedlm("resM$(kk)",XM)
writedlm("resR$(kk)",XR)
end
