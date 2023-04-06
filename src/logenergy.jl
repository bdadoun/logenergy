## Libraries
import ArnoldiMethod: partialschur
import Base.Threads: nthreads, @threads # for multi-threading
import Distributions: Chi, Gamma, quantile
import GenericLinearAlgebra: eigen
import Formatting: printfmt
import LinearAlgebra: tril
import Random: AbstractRNG, GLOBAL_RNG, seed!, shuffle!
import SparseArrays: sparse, spzeros
import SpecialFunctions: gamma
import Statistics: mean

# double precision
using Quadmath

# Argument parsing
import ArgParse: ArgParseSettings, add_arg_table, parse_args

# Syntactic sugar to get eigenvalues for both sparse and full matrices
eig(m) = eigen(m).values
eig(m) = partialschur(m, nev=size(m, 1))[1].eigenvalues

## Generalized gamma sampling
randg(a::Real, d::Real, p::Real, args...) = randg(GLOBAL_RNG, Float64, a, d, p, args...)
randg(T::Type{<:AbstractFloat}, a::Real, d::Real, p::Real, args...) = randg(GLOBAL_RNG, T, a, d, p, args...)
randg(rng::AbstractRNG, a::Real, d::Real, p::Real, args...) = randg(rng, Float64, a, d, p, args...)
function randg(rng::AbstractRNG, T::Type{<:AbstractFloat}, a::Real, d::Real, p::Real, args...)
    return a * map(x -> quantile(Gamma(d/p, 1), x), rand(rng, T, args...)) .^ (1/p)
end

## Transforms
# Complex transform
function complexTransform(model::Function)
    return (model() + im*model()) / sqrt(2)
end

# Wigner transform
function wignerTransform(model::Function)
    m = model()
    if typeof(m[1, 1]) <: Complex
        return (m + m') / sqrt(2)
    else
        return tril(m,) + tril(m, -1)'
    end
end

# Wishart transform
function wishartTransform(model::Function, rho::Integer=1)
    m = model()
    for i in range(1, rho)
        m = hcat(m, model())
    end
    return m * m' / rho
end

# Sparsification
sparsify(m::Matrix, s::Real=1//2) = sparsify(GLOBAL_RNG, m, s)
function sparsify(rng::AbstractRNG, m::Matrix, s::Real=1//2)
    return m .* (rand(rng, size(m)...) .< 1 - s) / sqrt(1 - s)
end

## Full matrix models
# Ginibre Ensemble
gaussianEnsemble(n::Integer=32) = gaussianEnsemble(GLOBAL_RNG, Float64, n)
gaussianEnsemble(rng::AbstractRNG=GLOBAL_RNG, n::Integer=32) = gaussianEnsemble(rng, Float64, n)
gaussianEnsemble(T::Type{<:AbstractFloat}=Float64, n::Integer=32) = gaussianEnsemble(GLOBAL_RNG, T, n)
function gaussianEnsemble(rng::AbstractRNG=GLOBAL_RNG, T::Type{<:AbstractFloat}=Float64, n::Integer=32)
    return randn(rng, T, n, n) / sqrt(n)
end

# Rademacher Ensemble
rademacherEnsemble(n::Integer=32) = rademacherEnsemble(GLOBAL_RNG, Float64, n)
rademacherEnsemble(rng::AbstractRNG=GLOBAL_RNG, n::Integer=32) = rademacherEnsemble(rng, Float64, n)
rademacherEnsemble(T::Type{<:AbstractFloat}=Float64, n::Integer=32) = rademacherEnsemble(GLOBAL_RNG, T, n)
function rademacherEnsemble(rng::AbstractRNG=GLOBAL_RNG, T::Type{<:AbstractFloat}=Float64, n::Integer=32)
    return (2*(rand(rng, T, n, n).< T(.5)) .- T(1)) / sqrt(n)
end

# Uniform Ensemble
uniformEnsemble(n::Integer=32) = uniformEnsemble(GLOBAL_RNG, Float64, n)
uniformEnsemble(rng::AbstractRNG=GLOBAL_RNG, n::Integer=32) = uniformEnsemble(rng, Float64, n)
uniformEnsemble(T::Type{<:AbstractFloat}=Float64, n::Integer=32) = uniformEnsemble(GLOBAL_RNG, T, n)
function uniformEnsemble(rng::AbstractRNG=GLOBAL_RNG, T::Type{<:AbstractFloat}=Float64, n::Integer=32)
    return (2*rand(rng, T, n, n) .- T(1)) * sqrt(3/n)
end

# Heavy Ensemble
heavyEnsemble(n::Integer=32, q::Real=3) = heavyEnsemble(GLOBAL_RNG, Float64, n, q)
heavyEnsemble(rng::AbstractRNG=GLOBAL_RNG, n::Integer=32, q::Real=3) = heavyEnsemble(rng, Float64, n, q)
heavyEnsemble(T::Type{<:AbstractFloat}=Float64, n::Integer=32, q::Real=3) = heavyEnsemble(GLOBAL_RNG, T, n, q)
function heavyEnsemble(rng::AbstractRNG=GLOBAL_RNG, T::Type{<:AbstractFloat}=Float64, n::Integer=32, q::Real=3)
    return rademacherEnsemble(rng, T, n) .* (rand(rng, T, n, n).^(-1/q) .- T(1)) * sqrt((q-1)*(q-2)/2)
end

# Symmetric Generalized Gamma Ensemble
symmetricGeneralizedGammaEnsemble(n::Integer=32, a::Real=1.0, d::Real=1.0, p::Real=2.0) = symmetricGeneralizedGammaEnsemble(GLOBAL_RNG, Float64, n, a, d, p)
symmetricGeneralizedGammaEnsemble(T::Type{<:AbstractFloat}=Float64, n::Integer=32, a::Real=1.0, d::Real=1.0, p::Real=2.0) = symmetricGeneralizedGammaEnsemble(GLOBAL_RNG, Float64, n, a, d, p)
symmetricGeneralizedGammaEnsemble(rng::AbstractRNG, n::Integer=32, a::Real=1.0, d::Real=1.0, p::Real=2.0) = symmetricGeneralizedGammaEnsemble(rng, Float64, n, a, d, p)
function symmetricGeneralizedGammaEnsemble(rng::AbstractRNG=GLOBAL_RNG, T::Type{<:AbstractFloat}=Float64, n::Integer=32, a::Real=1.0, d::Real=1.0, p::Real=2.0)
    m = randg(rng, T, a, d, p, n, 2*n)
    m = m[:, 1:n] - m[:, (n+1):(2*n)]
    return m / sqrt(2*n*a^2*(gamma((d+2)/p)/gamma(d/p) - (gamma((d+1)/p)/gamma(d/p))^2))
end

## Sparse matrix models
regularGraph(n::Integer=32, k::Integer=3) = regularGraph(GLOBAL_RNG, n, k)
function regularGraph(rng::AbstractRNG=GLOBAL_RNG, n::Integer=32, k::Integer=3)
    # need n*k even and n ≥ k+1
    m = []
    d = 0
    edges = [[i j] for i in 1:(n-1) for j in (i+1):n]
    while ~all(d .== k)
        m = spzeros(Int, n, n)
        d = zeros(1, n)
        shuffle!(edges)
        e = copy(edges)
        while ~isempty(e)
            u, v = pop!(e)
            if m[u, v] == 0 && d[u] < k && d[v] < k
                m[u, v] = 1
                m[v, u] = 1
                d[u] += 1
                d[v] += 1
            end
        end
    end
    return m
end

# beta-Hermite Ensemble
hermiteEnsemble(n::Integer=32, beta::Real=2.0) = hermiteEnsemble(GLOBAL_RNG, n, beta)
function hermiteEnsemble(rng::AbstractRNG=GLOBAL_RNG, n::Integer=32, beta::Real=2.0)
    s = map(x -> rand(rng, Chi(x)), beta * ((n-1) : -1: 1))
    return sparse(
        [1:n; 2:n; 1:(n-1)],
        [1:n; 1:(n-1); 2:n],
        [sqrt(2) * randn(n); s; s]) / sqrt(2 + beta*(n-1)
    )
end

# beta-Laguerre Ensemble
laguerreEnsemble(n::Integer=32, beta::Real=2.0, rho::Integer=1) = laguerreEnsemble(GLOBAL_RNG, n, beta, rho)
function laguerreEnsemble(rng::AbstractRNG=GLOBAL_RNG, n::Integer=32, beta::Real=2.0, rho::Integer=1)
    k = n * rho
    m = sparse([1:n; 2:n], [1:n; 1:(n-1)],
        [map(x -> rand(rng, Chi(x)), beta * (k : -1 : (k-n+1)));
         map(x -> rand(rng, Chi(x)), beta * ((n-1) : -1 : 1))])
    return m * m' / (beta * k)
end

## The log-energy functional
function logEnergy(model::Function, components)
    a, b, c = 0, 0, [0]
    while ~all(c .!= 0)
        a = eig(model())
        b = eig(model())
        c = b .- a'
    end
    return [-mean(log.(abs.(c))) map(f -> mean((f.(a) .+ f.(b))/2), components)]
end

## Main program
function main()
    # All models
    models = [
        (rademacherEnsemble, true, "+/-")
        (gaussianEnsemble, true, "gaussian")
        (uniformEnsemble, true, "uniform")
        (heavyEnsemble, true, "heavy-tailed")
        (symmetricGeneralizedGammaEnsemble, true, "symmetric generalized Gamma")
        (hermiteEnsemble, false, "Tridiagonal beta-Hermite matrix model")
        (laguerreEnsemble, false, "Tridiagonal beta-Laguerre matrix model")
        (regularGraph, false, "Regular Graph model")
    ]

    arg_desc = Dict(
        "n" => "Matrix dimension (default = 32)",
        "beta" => "Inverse temperature (default = 2.0)",
        "a" => "Scale parameter of the generalized gamma distribution (default = 1.0)",
        "d" => "First shape parameter of the generalized gamma distribution (default = 1.0)",
        "p" => "Second shape parameter of the generalized gamma distribution (default = 2.0)",
        "q" => "Weight parameter (default = 3.0)",
        "s" => "Sparsification threshold (default = 0)",
        "rho" => "Rectangular #columns/#rows integer ratio",
        "k" => "Degree of the regular graph",
        "T" => "Precision type (default = Float64)",
        "rng" => "Random Number Generator (default = GLOBAL_RNG)"
    )

    # Command line settings
    settings = ArgParseSettings()
    commands = Dict()
    for (f, full, desc) in models
        name = replace(string(f), r"Ensemble" => "")
        if length(name) > 10
            name = replace(name, r"(?<=[A-Za-z])[a-z]" => "")
        end
        commands[name] = Dict("func" => f, "args" => [])
        if full
            desc = "Full random matrix model with $(desc) coefficients"
        end
        add_arg_table(settings,
            name,
            Dict(
                :action => "command",
                :help => desc * "."
            )
        )
        for match in eachmatch(r"([^,(\s]+)::([^,)\s]+)", string(last(methods(f))))
            arg = match[1]
            type = eval(Meta.parse(match[2]))
            push!(commands[name]["args"], (arg, type))
            add_arg_table(settings[name],
                (length(match[1]) == 1 ? "-" * match[1] : "--" * match[1]),
                Dict(
                    :required => false,
                    :help => "$(arg_desc[arg])"
                )
            )
        end
        if full
            add_arg_table(settings[name],
                ["--complex", "-c"],
                Dict(
                    :action => :store_true,
                    :help => "Sample the coefficients in the complex field."
                ),
                ["--sparsify", "-s"],
                Dict(
                    :arg_type => Float64,
                    :constant => 0.5,
                    :default => 0.0,
                    :help => "Sparsify the matrix using the given sparsification parameter.",
                    :nargs => '?'
                ),
                ["--wigner"],
                Dict(
                    :action => :store_true,
                    :help => "Sample the additive hermitization of the matrix model.",
                ),
                ["--wishart"],
                Dict(
                    :arg_type => Int,
                    :constant => 1,
                    :default => nothing,
                    :help => "Sample the multiplicative hermitization of the matrix model.",
                    :nargs => '?'
                )
            )
        end
    end

    # Random seed
    add_arg_table(settings, "--seed", Dict(
        :arg_type => Int,
        :default => 2023,
        :help => "Set the random seed (default = 2023).",
    ))

    # Monte Carlo cycles
    add_arg_table(settings, "--cycles", Dict(
        :arg_type => Int,
        :default => 1000,
        :help => "Set the number of Monte Carlo cycles to perform (default = 1000)."
    ))

    # Option to print an example
    add_arg_table(settings, "--example", Dict(
        :action => :store_true,
        :help => "Prints an example matrix with the output."
    ))

    # Parse command line arguments
    args = parse_args(settings)
    model = args["%COMMAND%"]
    jobname = model
    opts = args[model]
    f = commands[model]["func"]
    f_args = []
    for (arg, type) in commands[model]["args"]
        if args[model][arg] != nothing
            val = eval(Meta.parse(args[model][arg]))
            if ~isa(val, type)
                println(stderr, "Warning: input -", arg, " ", val, " ignored as it is not of type ", type, ".")
            else
                push!(f_args, val)
                jobname *= "-" * string(val)
            end
        end
    end

    # Components of the log-energy functional
    moment1 = ("moment1", identity)
    moment2 = ("moment2", x -> abs(x)^2)
    logmoment = ("logmoment", log)

    # Prepare the function to be averaged
    func = () -> f(f_args...)
    i = 1
    comp = [moment2]
    if haskey(opts, "complex") && opts["complex"]
        f1 = func
        func = () -> complexTransform(f1)
        jobname *= "-complex"
    end
    if haskey(opts, "sparsify") && opts["sparsify"] > 0
        f2 = func
        func = () -> sparsify(f2(), opts["sparsify"])
        jobname *= "-sp" * string(opts["sparsify"])
    end
    if haskey(opts, "wigner") && opts["wigner"]
        if opts["wishart"] != nothing
            println(stderr, "Warning: --wigner and --wishart are not compatible; ignoring --wishart.")
        end
        f3 = func
        func = () -> wignerTransform(f3)
        jobname *= "-wigner"
    elseif haskey(opts, "wishart") && opts["wishart"] != nothing
        f4 = func
        func = () -> wishartTransform(f4, opts["wishart"])
        jobname *= "-wishart"
        comp = [moment1]
    end
    if model == "laguerre"
        comp = [moment1, logmoment]
    end

    # Print job information
    cycles = args["cycles"]
    println("# model = $model")
    println("# jobname = $jobname")
    println("# seed = ", args["seed"])
    println("# cycles = $cycles")
    println("# threads = ", nthreads())
    println("# output = ", join(["log-entropy"; map(y -> y[1], comp)], ";"))

    seed!(args["seed"])

    # Print an example
    if args["example"]
        println("# example_matrix =")
        io = IOBuffer()
        show(IOContext(io), "text/plain", func())
        example = String(take!(io))
        println(replace(example, r"^"m => "#   "))
    end
    
    # Parallel computing
    comp = map(y -> y[2], comp)
    z = logEnergy(func, comp)
    data = repeat(z, cycles)
    @threads for i = 2 : cycles
        data[i, :] = logEnergy(func, comp)
    end
    x = cycles > 1 ? mean(data, dims=1) : data
    format = replace("{};" ^ length(x), r";$" => "\n")
    printfmt(format, x...)
end

main()