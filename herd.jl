include("./dbscan.jl")
include("./herd_functions.jl")

parameters_dict = Dict(
    "total_steps" => 1e7,          #Number of steps in the simulation
    "anim_steps" => 5e2,          #Number of steps in the animation
    "nₚₐᵣₜₛ" => 70,               #Maximum number of particles in the simulation
    "avg_repr" => false,           #Crossover reproduction (true) vs Select best "genes"
    "roulette" => false,           #Kill boids with weighted probabilities -> BUG FIX ROULETTE
    "weighted_repr" => false,      #Reproduction has a chance to happen depending of fitness difference
    "reflect_walls" => false,      #Confine boids in a closed space(true) vs Periodic boundary condition (false)
    "vision" => false,             #Boids can only perceive other boids at a certain direction
    "crossover" => true,
    "potential" => false,          #Boids are penalized for touching each other
    "innertia" => false,           #Boids have rotational innertia(true)
    "fermionic" => false,          #Boids have a hard core(true)
    "box_size" => 1,               #Size of the reservoir where boids are in
    "ε" => 0.5,                    #DB scan parameter
    "min_pts" => 3,                #DB minimum number of points per cluster
    "μ" => 1e-6,                   #Rate of mutation
    "β" => 25,                    #Boids vision distance center_of_mass .+= ((exp(- "β" *dist^2.0)/length(Boids)) * fov ) .* boid₂.pos
    "γ" => 10.0,                    #Fitness multiplier
    "κ" => 50.0,                   #S curve to avoid boids touching coefficient 1/(1 - exp(-κ(x-x0)))
    "x₀" => 0.07,                  #S curve to avoid boids touching center value 1/(1 - exp(-κ(x-x0)))
    "r" => 0.1,                    #Boid Interaction Radius
    "rr" => 0.03,                   #Boid Reproduction Radius
    "cr" => 0.02,                  #Boid Collision Radius
    "cn" => 0,                     #Boid Child Number
    "neighbors" => Int32[],        #Boid Initial neighbors
    "speed" => 1.0,                #Boid Velocity
    "τ" => 1,                      #Boid Lifetime
    "Δt" => 1e-2                   #
)

function update_boids!(parameters_dict::Dict)
    nₚₐᵣₜₛ = parameters_dict["nₚₐᵣₜₛ"]
    total_steps = parameters_dict["total_steps"]

    d_params = [parameters_dict["nₚₐᵣₜₛ"],
                 parameters_dict["total_steps"],
                 parameters_dict["Δt"],
                 parameters_dict["μ"],
                 parameters_dict["β"],
                 parameters_dict["κ"],
                 parameters_dict["γ"],
                 parameters_dict["x₀"],
                 parameters_dict["box_size"],
                 parameters_dict["vision"],
                 parameters_dict["innertia"],
                 parameters_dict["avg_repr"],
                 parameters_dict["roulette"],
                 parameters_dict["potential"],
                 parameters_dict["fermionic"],
                 parameters_dict["weighted_repr"],
                 parameters_dict["reflect_walls"],
                 parameters_dict["crossover"],
                ] |> cu

    d_r = Array{Float32}(undef, nₚₐᵣₜₛ)
    d_τ = Array{Int32}(undef, nₚₐᵣₜₛ)
    d_ϕ = Array{Float32}(undef, nₚₐᵣₜₛ)
    d_cn = Array{Int32}(undef, nₚₐᵣₜₛ)
    d_sx = Array{Bool}(undef, nₚₐᵣₜₛ)
    d_rr = Array{Float32}(undef, nₚₐᵣₜₛ)
    d_pos = Array{Float32, 2}(undef, 2, nₚₐᵣₜₛ)  # Assuming pos is a 2D array
    d_θ = Array{Float32}(undef, nₚₐᵣₜₛ)
    d_model = Vector{Chain}(undef, nₚₐᵣₜₛ)
    d_speed = Array{Float32}(undef, nₚₐᵣₜₛ)
    d_neighbors = Array{Int32, 2}(undef, 8, nₚₐᵣₜₛ)

    for idx in 1:nₚₐᵣₜₛ
        d_r[idx] = parameters_dict["r"]
        d_τ[idx] = parameters_dict["τ"]
        d_ϕ[idx] = Float32(rand())
        d_cn[idx] = parameters_dict["cn"]
        d_sx[idx] = rand(Bool)
        d_rr[idx] = parameters_dict["rr"]
        d_pos[:, idx] .= Float32.([2*rand()-1.0, 2*rand()-1.0])
        d_θ[idx] = Float32(π * rand(-1.0:1e-5:1.0))
        d_speed[idx] = parameters_dict["speed"]
    end

    d_r = d_r |> cu
    d_τ = d_τ |> cu
    d_ϕ = d_ϕ |> cu
    d_θ = d_θ |> cu
    d_cn = d_cn |> cu
    d_sx = d_sx |> cu
    d_rr = d_rr |> cu
    d_pos = d_pos |> cu
    d_speed = d_speed |> cu

    inputₗₑₙ = 3
    hiddenₗₑₙ = 3
    outₗₑₙ = 1

    W₁ = randn(Float32, nₚₐᵣₜₛ, hiddenₗₑₙ, inputₗₑₙ) * sqrt(2 / (hiddenₗₑₙ + inputₗₑₙ)) |> cu
    W₂ = randn(Float32, nₚₐᵣₜₛ, outₗₑₙ, hiddenₗₑₙ) * sqrt(2 / (outₗₑₙ + hiddenₗₑₙ)) |> cu

    b₁ = randn(Float32, nₚₐᵣₜₛ, hiddenₗₑₙ) |> cu
    b₂ = randn(Float32, nₚₐᵣₜₛ, outₗₑₙ) |> cu

    out = zeros(Float32, nₚₐᵣₜₛ, 3) |> cu
    x = zeros(Float32, nₚₐᵣₜₛ, hiddenₗₑₙ) |> cu
    avg_coord = zeros(Float32, nₚₐᵣₜₛ, 3) |> cu

    measure_every = 1e3
    t = [1:total_steps/measure_every]
    ψ = []
    ω = []
    Δ = []

    for τ in 1:total_steps
        @cuda threads=nₚₐᵣₜₛ update_kernel!(d_params, d_τ, d_ϕ, d_cn, d_sx, d_rr, d_pos, d_θ, d_speed, W₁, W₂, b₁, b₂, x, out, avg_coord)

        if τ % 1000 == 0
            println("step: $τ of $total_steps")
        end

        if τ % measure_every == 0
            ψᵢ,Δᵢ = order_parameter(nₚₐᵣₜₛ, d_pos|>cpu, d_θ|>cpu, d_speed|>cpu)
            ωᵢ = calculate_vorticity(parameters_dict,nₚₐᵣₜₛ, d_pos|>cpu, d_θ|>cpu)
            push!(ψ,ψᵢ)
            push!(Δ,Δᵢ)
            push!(ω,ωᵢ)
#             println((W₁ |> cpu)[1,:,:])
        end
    end

    println("Starting Animation")
    anim_size = parameters_dict["anim_steps"]
    anim = @animate for τ in 1:anim_size
        @cuda threads=nₚₐᵣₜₛ update_kernel!(d_params, d_τ, d_ϕ, d_cn, d_sx, d_rr, d_pos, d_θ, d_speed, W₁, W₂, b₁, b₂, x, out, avg_coord)

        θ = d_θ |> cpu
        pos = d_pos |> cpu
#         W₁ |> cpu
#         W₂ |> cpu
#         b₁ |> cpu
#         b₂ |> cpu

        Plots.scatter(pos[1,:], pos[2,:], aspect_ratio=:equal, legend=false)
        quiver!(pos[1,:], pos[2,:], quiver=(0.02*cos.(θ), 0.02*sin.(θ)), color=:blue)

        xlims!(0, parameters_dict["box_size"])
        ylims!(0, parameters_dict["box_size"])

        if τ % 100 == 0
            println("animation step: $τ of $anim_size")
        end
    end

    μ = parameters_dict["μ"]
    β = parameters_dict["β"]
    γ = parameters_dict["γ"]
    vision = parameters_dict["vision"]
    innertia = parameters_dict["innertia"]
    roulette = parameters_dict["roulette"]
    crossover = parameters_dict["crossover"]

    folder_path = "./results/steps:$total_steps-n_parts:$nₚₐᵣₜₛ-vision:$vision-innertia:$innertia-roulette:$roulette-crossover:$crossover-mutation:$μ"

    if !isdir(folder_path)
        # Create the folder if it does not exist
        mkdir(folder_path)
        println("Folder created: $folder_path")
    else
        println("Folder already exists: $folder_path")
    end

    gif(anim, "$folder_path/boid_animation.mp4", fps = 10)

    plot(t, ψ, xlims=(1,Inf), label="Order Parameter Over Time")
    savefig("$folder_path/order_param.png")

    plot(t, Δ, xlims=(1,Inf), label="Average Distance Over Time")
    savefig("$folder_path/avg_dist.png")

    println(ω)
    plot(t, ω, label="Circulation Over Time")
    savefig("$folder_path/circulation.png")
end

update_boids!(parameters_dict)
