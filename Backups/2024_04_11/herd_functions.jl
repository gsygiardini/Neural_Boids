using CSV
using CUDA
using FFTW
using Flux
using BSON
using Plots
using StatsBase
using DataFrames
using Statistics
using Colors, ColorSchemes, PerceptualColourMaps
using LinearAlgebra


mutable struct Boid <: Function
    model::Chain              #Boids Neural Network model
    pos::Vector{Float32}   #Boid Position
    θ::Float32                #Boid Orientation
    speed::Float32            #Boid Velocity
    ϕ::Float32                #Boid's Fitness (Child Number Over time)
    τ::Int32                  #Boid's Lifetime
    r::Float32                #Interaction Radius
    sx::Bool                  #Boid "Sex"
    rr::Float32               #Boid's Reproduction radius
    cn::Int32                 #Boid's Child Number
    index::Int32              #Boid Index
    neighbors::Array{Int32} #Neighbor List

    function Boid(parameters_dict::Dict,
        model::Chain,
        pos::Vector{Float32},
        θ::Float32=Float32(π * rand(-1.0:1e-5:1.0)),               # Boid Orientation# =[2*rand()-1.0, 2*rand()-1.0],  # Boid Position
        ϕ::Float32=Float32(rand()),
        sx::Bool=rand(Bool)                                # Boid "sex"
        )

        index = 1
        τ = Int64(parameters_dict["τ"])  # Convert to Int64
        cn = Int64(parameters_dict["cn"])  # Convert to Int64

        r = Float64(parameters_dict["r"])  # Convert to Float64
        rr = Float64(parameters_dict["rr"])  # Convert to Float64
        speed = Float64(parameters_dict["speed"])  # Convert to Float64

#         pos = reshape(CUDA.CuArray(pos), :, 1)

        neighbors = parameters_dict["neighbors"]

        new(model, pos, θ, speed, ϕ, τ, r, sx, rr, cn, index, neighbors)
    end
end

function create_boids(parameters_dict::Dict,n_particles::Int64)
    Boids = Boid[]

    for i in 1:n_particles
        model = Chain(Dense(3, 3, tanh; init=Flux.glorot_uniform(gain=sqrt(2))),
                Dense(3, 2, tanh; init=Flux.glorot_uniform(gain=sqrt(2))),
                Dense(2, 1, tanh; init=Flux.glorot_uniform(gain=1))) |> cu

        pos = Float32.([2*rand()-1.0, 2*rand()-1.0])
        boid = Boid(parameters_dict, model, pos)
        push!(Boids,boid)
    end

    return Boids
end

function reproduction(parameters_dict::Dict,Boids::Vector{Boid})
    updated_Boids = Boids
    Nₚₐᵣₜₛ = length(Boids)

    avg = parameters_dict["avg_repr"]
    maxₚₐᵣₜₛ = parameters_dict["maxₚₐᵣₜₛ"]
    weighted = parameters_dict["weighted_repr"]

    for (i, boid) in enumerate(Boids)
        for (j, neighbor_idx) in enumerate(boid.neighbors)
            if neighbor_idx > i
                neighbor = Boids[neighbor_idx]

                prob = 0.0
                if weighted==true
                    ρ=0.01
                    #Probability of Mating. Here, the higher the greater the chance
                    #prob = 2.0*(exp(ρ*abs(boid.ϕ + neighbor.ϕ))/(1+exp(ρ*abs(boid.ϕ + neighbor.ϕ))) - 0.5)
                    #Probability of Mating. Here, the smaller the difference between fitness the greater the chance
                    prob = 2.0*(exp(ρ*abs(boid.ϕ - neighbor.ϕ))/(1+exp(ρ*abs(boid.ϕ - neighbor.ϕ))) - 0.5)
                end

                if Nₚₐᵣₜₛ > maxₚₐᵣₜₛ
                    if norm(boid.pos - neighbor.pos) < boid.rr && boid.sx != neighbor.sx && rand(0.0:1.0) >= prob
                        if avg == true
                            model = crossover(parameters_dict, boid, neighbor)
                        else
                            model = best_params(parameters_dict, boid, neighbor)
                        end

                        boid.cn+=1
                        neighbor.cn+=1

                        killed_Boid, updated_Boids = death(parameters_dict, updated_Boids)

                        #Boid Child Number
                        push!(updated_Boids,
                            Boid(parameters_dict,
                                model,
                                killed_Boid.pos,        #Boids are born at the position of a boid that died
                                )
                            )
                        break
                    end
                else
                    if norm(boid.pos - neighbor.pos) < boid.rr && boid.sx != neighbor.sx && rand(0.0:1.0) >= prob
                        if avg == true
                            model = crossover(parameters_dict, boid, neighbor)
                        else
                            model = best_params(parameters_dict, boid, neighbor)
                        end

                        boid.cn+=1
                        neighbor.cn+=1

                        #Boid Child Number
                        push!(updated_Boids,
                            Boid(parameters_dict,
                                model,
                                [2*rand()-1.0, 2*rand()-1.0]     #Boids are born at a random position
#                                 (boid.pos + neighbor.pos)/2.0, #Boids are born between the parents (leads to premature convergence)
                                )
                            )
                        break
                    end
                end
            end
        end
    end

    return updated_Boids
end

function death(parameters_dict::Dict, Boids::Vector{Boid})
    roulette = parameters_dict["roulette"]

    killed_Boids = Boid[]
    updated_Boids = Boid[]
    min_ϕ = Inf


    if roulette==true
        prob = []
        total_ϕ = sum([b.ϕ for b in Boids])

        for boid in Boids
            push!(prob,(total_ϕ-boid.ϕ))
        end

        while Boids[i].τ==0
            i = 0
            r = rand()*total_ϕ
            while(r ≥ 0)
                i += 1
                r -= prob[i]
            end
        end

        killed_Boid = Boids[i]
        deleteat!(Boids,i)
    else
        for boid in Boids
            if boid.τ > 0
                min_ϕ = min(min_ϕ, boid.ϕ)
            end
        end

        for boid in Boids
            if boid.τ == 0
                push!(updated_Boids, boid)
            elseif boid.ϕ > min_ϕ
                push!(updated_Boids, boid)
            else
                push!(killed_Boids, boid)
            end
        end

        if !isempty(killed_Boids)
            random_kill = rand(1:length(killed_Boids))

            killed_Boid = killed_Boids[random_kill]

            append!(updated_Boids, deleteat!(killed_Boids, random_kill))
        else
            killed_Boid = 0
        end

        Boids = copy(updated_Boids)
    end

    return killed_Boid, Boids
end

function update_boids!(Boids::Vector{Boid}, parameters_dict::Dict, Δt::Float64)

    nₚₐᵣₜₛ = 10#parameters_dict["maxₚₐᵣₜₛ"]


#     d_params = Int32.([parameters_dict["vision"],
#               parameters_dict["innertia"],
#               parameters_dict["avg_repr"],
#               parameters_dict["roulette"],
#               parameters_dict["potential"],
#               parameters_dict["fermionic"],
#               parameters_dict["weighted_repr"],
#               parameters_dict["reflect_walls"],
#               ]) |> cu
    # Transfer data to GPU memory

    # Initialize CuArrays directly
    d_i = Array{Int32}(undef, length(Boids))
    d_r = Array{Float32}(undef, length(Boids))
    d_τ = Array{Int32}(undef, length(Boids))
    d_ϕ = Array{Float32}(undef, length(Boids))
    d_cn = Array{Int32}(undef, length(Boids))
    d_sx = Array{Bool}(undef, length(Boids))
    d_rr = Array{Float32}(undef, length(Boids))
    d_pos = Array{Float32, 2}(undef, 2, length(Boids))  # Assuming pos is a 2D array
    d_coords = Array{Float32, 2}(undef, 3, length(Boids))  # Assuming pos is a 2D array
    d_θ = Array{Float32}(undef, length(Boids))
    d_model = Vector{Chain}(undef, length(Boids))
    d_speed = Array{Float32}(undef, length(Boids))
#     d_neighbors = Vector{Int32}(undef, 8, length(Boids))

    # Fill the CuArrays with data from Boids
    for (i, boid) in enumerate(Boids)
        d_i[i] = boid.index
        d_r[i] = boid.r
        d_τ[i] = boid.τ
        d_ϕ[i] = boid.ϕ
        d_cn[i] = boid.cn
        d_sx[i] = boid.sx
        d_rr[i] = boid.rr
        d_pos[:, i] .= boid.pos
        d_coords[:,i] .= vcat(boid.pos,[boid.θ])
        d_θ[i] = boid.θ
        d_model[i] = boid.model
        d_speed[i] = boid.speed
    end

    inputₗₑₙ = 3
    hiddenₗₑₙ = 3
    outₗₑₙ = 1

    W₁ = randn(Float32, nₚₐᵣₜₛ, hiddenₗₑₙ, inputₗₑₙ) |> cu
    W₂ = randn(Float32, nₚₐᵣₜₛ, outₗₑₙ, hiddenₗₑₙ) |> cu

    b₁ = randn(Float32, nₚₐᵣₜₛ, hiddenₗₑₙ) |> cu
    b₂ = randn(Float32, nₚₐᵣₜₛ, outₗₑₙ) |> cu

    cm = zeros(Float32, 2, 2) |> cu

    x = zeros(Float32, inputₗₑₙ) |> cu

    out = zeros(Float32, 3) |> cu

    avg_coord = zeros(Float32, 3) |> cu

    # Launch CUDA kernel
#     @cuda threads=length(Boids) update_kernel!(d_i|>cu, d_r|>cu, d_τ|>cu, d_ϕ|>cu, d_cn|>cu, d_sx|>cu, d_rr|>cu, d_pos|>cu, d_angle|>cu, d_model)#, d_speed, d_neighbors, d_params)
    @cuda threads=length(Boids) update_kernel!(d_i|>cu, d_r|>cu, d_θ|>cu, d_τ|>cu, d_ϕ|>cu, d_cn|>cu, d_sx|>cu, d_rr|>cu, d_coords|>cu, d_speed|>cu#=, d_params|>cu=#, W₁, W₂, b₁, b₂, cm, x, out, avg_coord)

#     # Transfer updated data back to CPU memory
#     CUDA.copyto!(updated_Boids, d_updated_Boids)
end

function update_kernel!(d_i, d_r, d_θ, d_τ, d_ϕ, d_cn, d_sx, d_rr, d_coords, d_speed#=, d_params=#, W₁, W₂, b₁, b₂, cm, x, out, avg_coord)
    idx = threadIdx().x + (blockIdx().x - 1) * blockDim().x

    box_size=1
    Δt = 1e-2
    β = 1.0
    avg_θ = 0.0

    #Get direction from neural network
    ############################################################################################################################
    for jdx in 1:length(d_i)
        if idx != jdx
            dist = 0.0
            @inbounds fov = (1.0 + cos(d_coords[3,idx] - d_coords[3,jdx])) #Field of vision
            for kdx in 1:2
                @inbounds dist += (d_coords[kdx,idx] - d_coords[kdx,jdx])^2
            end
            dist = sqrt(dist)
            @inbounds avg_θ += ((exp(-β*dist^2.0)/length(d_i))) * fov * d_coords[3,jdx]

            for kdx in 1:2
                @inbounds avg_coord[kdx] += ((exp(-β*dist^2)/length(d_i)) * fov) * d_coords[kdx,jdx]
            end

#             #d_neighbors[idx] = jdx
        end
    end

    x[1] = avg_coord[1]
    x[2] = avg_coord[2]
    x[3] = avg_θ

    for jdx in 1:size(W₁,3)
        tmp = 0
        for kdx in 1:size(W₁,2)
            @inbounds tmp += W₁[idx,kdx,jdx] * x[kdx]
        end
        out[jdx] = tmp
    end

    for jdx in 1:size(W₂,3)
        tmp = 0
        for kdx in 1:size(W₂,2)
            @inbounds tmp += W₂[idx,kdx,jdx] * tanh(out[kdx] + b₁[idx,kdx])
        end
        out[jdx] = tmp
    end
    @inbounds out .= out .+ b₂[idx]
    d_θ[idx] = π*out[size(W₂,3)]
    ############################################################################################################################

    #Recalculate particles' coordinates x,y,θ
    ############################################################################################################################
    if innertia
        d_coords[3,idx] = mod(d_coords[3,idx] + d_θ[idx]* Δt + π, 2.0 * π) - π
    else
        d_coords[3,idx] = d_θ[idx]
    end

    d_coords[1,idx] = d_coords[1,idx] + d_speed[idx] * Δt * cos(d_coords[3,idx])
    d_coords[2,idx] = d_coords[2,idx] + d_speed[idx] * Δt * sin(d_coords[3,idx])
    ############################################################################################################################


    #Create boundary conditions
    ############################################################################################################################
    d_coords[1,idx] = mod(d_coords[1,idx] + box_size, box_size)
    d_coords[2,idx] = mod(d_coords[2,idx] + box_size, box_size)

    ############################################################################################################################
    
    

#
#         if fermionic
#             volume_exclusion!(boid₁, updated_Boids, Δt)
#         end
#
#         boid₁.pos[1], boid₁.pos[2], boid₁.θ = boundary_condition(parameters_dict, boid₁.pos[1], boid₁.pos[2], boid₁.θ)
#         updated_Boids[index] = deepcopy(boid₁)
#     end

    return
end
#=

function simu!(parameters_dict::Dict, Boids::Vector{Boid}, Δt::Float64)
    index = threadIdx().x + (blockIdx().x - 1) * blockDim().x

    updated_Boids = copy(Boids)
    distances = zeros(Float64,length(Boids),length(Boids))
    innertia = parameters_dict["innertia"]
    fermionic = parameters_dict["fermionic"]

    for i in 1:length(Boids)
        boid₁ = deepcopy(Boids[i])

        neighbors = []

        for j in i:length(Boids)
            if i!=j
                boid₂ = Boids[j]
                if norm(boid₁.pos - boid₂.pos) < boid₁.r
                    push!(neighbors,j)
                end
            end
        end
        boid₁.neighbors = neighbors

        if innertia == true
            boid₁.θ = mod(updated_Boids[i].θ + get_direction(boid₁,Boids)*Δt + π, 2.0*π) - π
        else
            updated_Boids[i].θ = get_direction(parameters_dict,boid₁,Boids)
        end

        boid₁.pos .= boid₁.pos + boid₁.speed * Δt * [cos(boid₁.θ), sin(boid₁.θ)]

        if fermionic == true
            volume_exclusion!(boid₁,updated_Boids, Δt)
        end

        boid₁.pos[1],boid₁.pos[2],boid₁.θ = boundary_condition(parameters_dict,boid₁.pos[1],boid₁.pos[2],boid₁.θ)
        updated_Boids[i] = deepcopy(boid₁)
    end

    for boid in updated_Boids
        boid.τ += 1
    end

    Boids = reproduction(parameters_dict,updated_Boids)
end=#

function best_params(parameters_dict::Dict, boid₁::Boid, boid₂::Boid)
    model₁ = boid₁.model
    model₂ = boid₂.model

    λ = parameters_dict["λ"]

    if boid₁.ϕ >= boid₂.ϕ
        collected = mutate(collect(Flux.params(model₁)),λ)
    else
        collected = mutate(collect(Flux.params(model₂)),λ)
    end

    model = Chain(Dense(3, 3, tanh),
                Dense(3, 2, tanh),
                Dense(2, 1, tanh))

    Flux.loadparams!(model, collected)
    return model
end

function crossover(parameters_dict::Dict, boid₁::Boid, boid₂::Boid)
    model₁ = boid₁.model
    model₂ = boid₂.model
    @assert length(model₁.layers) == length(model₂.layers)  # Make sure models have the same structure

    λ = parameters_dict["λ"]

    # Collect parameters from each model anCd take the average
    collected1 = collect(Flux.params(model₁))
    collected2 = collect(Flux.params(model₂))
    collected = mutate([rand([p1, p2]) for (p1, p2) in zip(collected1, collected2)],λ)

    # Create a new model with averaged parameters
    averaged_model = deepcopy(model₁)  # Create a copy of model1
    Flux.loadparams!(averaged_model, collected)  # Load averaged parameters into the new model
    return averaged_model
end

function mutate(vec::Vector{Any}, λ::Float64)
    for i in 1:length(vec)
        if isa(vec[i], Array{Float32, 2})
            for j in 1:size(vec[i], 1)
                for k in 1:size(vec[i], 2)
                    vec[i][j, k] = (1-λ)*vec[i][j, k] + λ * rand(-1.0:1e-5:1.0)  # Modify each element using some function
                end
            end
        elseif isa(vec[i], Float32)
            vec[i] = (1-λ)*vec[i] + λ * rand(-1.0:1e-5:1.0)  # Modify element directly
        end
    end
    return vec
end

function fitness(parameters_dict::Dict, Boids::Vector{Boid}, τ::Int64)
    γ = parameters_dict["γ"]
    κ = parameters_dict["κ"]
    x₀ = parameters_dict["x₀"]
    potential = Inf

    for boid₁ in Boids
        for boid₂ in Boids
            if (boid₁ != boid₂)
                dist = norm(boid₁.pos - boid₂.pos)
                cost = 1 / (1 + exp(-κ*(dist-x₀)))
                potential = min(cost,potential)
            end
        end

        if boid₁.τ > 0
            offspring_rate = γ * boid₁.cn / boid₁.τ
        else
            offspring_rate = 0.0
        end

        boid₁.ϕ = offspring_rate * potential
    end
end

function volume_exclusion!(boid₁::Boid, Boids::Vector{Boid}, Δt::Float64)
    #Particles cant occupy same space

    for boid₂ in Boids
        dist = norm(boid₁.pos - boid₂.pos)
#         if dist <= (boid₁.cr + boid₂.cr)
#             collision+=1
#         end

        if boid₁ !== boid₂ && dist <= (boid₁.cr + boid₂.cr) / 2.0
            move_dir = normalize!(boid₁.pos - boid₂.pos)
            move_amount = ((boid₁.cr + boid₂.cr) - dist) / 2.0

            boid₁.pos += move_amount * move_dir
            boid₂.pos -= move_amount * move_dir
        end
    end
end

function reset_positions!(boids::Vector{Boid})
    for boid in boids
        boid.θ = π * rand(-1.0:1e-5:1.0)
        boid.pos = [2*rand()-1.0, 2*rand()-1.0]
    end
end

function boundary_condition(parameters_dict::Dict, x::Float64,y::Float64,θ::Float64)
    box_size=parameters_dict["box_size"]
    constrain=parameters_dict["reflect_walls"]

    if constrain==true
         # Reflective boundary conditions for x-coordinate
        if x <= 0.0
            x = abs(x)
            θ = π - θ
        elseif x >= box_size
            x = 2 * box_size - x
            θ = π - θ
        end

        # Reflective boundary conditions for y-coordinate
        if y <= 0.0
            y = abs(y)
            θ = -θ
        elseif y >= box_size
            y = 2 * box_size - y
            θ = 2.0*π-θ
        end
    else
        x = mod(x + box_size, box_size)
        y = mod(y + box_size, box_size)
        θ = θ
    end

    return x, y, θ
end

function order_parameter(Boids::Vector{Boid})
    v = [0,0]
    sum_speed = 0
    for boid in Boids
        v = v .+ boid.speed * [cos(boid.θ), sin(boid.θ)]
        sum_speed = sum_speed + boid.speed
    end

    avg_speed = sum_speed/length(Boids)

    ψ = norm(v)/(length(Boids)*avg_speed)

    return ψ
end

function collisions(Boids::Vector{Boid})
    #Number of particle collisions
    counts = 0
    distances = []

    for i in 1:length(Boids)
        for j in i+1:length(Boids)
            distance = norm(Boids[i].pos - Boids[j].pos)
            push!(distances,distance)
            if distance < Boids[i].cr
                counts+=1
            end
        end
    end

    return counts, mean(distances)
end

function save_models(Boids::Vector{Boid}, prefix="model", folder="./saved_models/")
    if !isdir(folder)
        mkdir(folder)
    end

    for (i, boid) in enumerate(Boids)
        model_filename = "$folder$prefix$i.bson"
        model = boid.model
        BSON.@save model_filename model
    end
end

# Function to load models
function load_models(Boids::Vector{Boid}, prefix="model", folder="./saved_models/")
    if !isdir(folder_path)
        println("No model was found")
        return 0
    end

    for (i, boid) in enumerate(Boids)
        model_filename = "$folder$prefix$i.bson"
        boid.model = BSON.load(model_filename)[:model]
    end
end

function db_scan(parameters_dict::Dict, Boids::Vector{Boid})
    ε = parameters_dict["ε"]
    min_pts = parameters_dict["min_pts"]

    models = []
    distances = []

    for boid in Boids
        model = []
        collected = collect(Flux.params(boid.model))
        for item in collected
            push!(model, cat(item..., dims=1))
        end
        push!(models, cat(model..., dims=1))
    end

    #Write down the dsitances if i!=j
    for i in 1:length(models)
        for j in 1:length(models)
            if i!=j
                push!(distances,euclidean_distance(models[i],models[j]))
            end
        end
    end

    hist = Plots.histogram(distances, bins=20)

    clusters = dbscan(models,ε,min_pts)

    return clusters, hist
end

# Function to flatten parameters of a Flux model into a 1D array
function flatten_params(model::Chain)
    return collect(Iterators.flatten(Flux.params(model)))
end

# Function to store models in a contiguous block of memory
function store_models(models::Vector{Chain})
    flat_params = Vector{Float32}()
    for model in models
        append!(flat_params, flatten_params(model))
    end
    return flat_params
end

# Function to get pointers to each model in the contiguous memory block
function get_model_pointers(flat_params::Vector{Float32}, models::Vector{Chain})
    model_pointers = Vector{Ptr{Float32}}()
    base_ptr = pointer(flat_params)
    for model in models
        push!(model_pointers, base_ptr)
        base_ptr += length(flatten_params(model))
    end
    return model_pointers
end

index_colors = Dict(
    -1 => :black,
    0 => :red,
    1 => :green,
    2 => :blue,
    3 => :yellow,
    4 => :purple,
    5 => :orange,
    6 => :cyan,
    7 => :magenta,
    8 => :brown,
    9 => :pink,
    10 => :teal,
    11 => :olive,
    12 => :navy,
    13 => :maroon,
    14 => :lime,
    15 => :skyblue,
    16 => :gold,
    17 => :indigo,
    18 => :tan,
    19 => :violet,
    20 => :salmon,
    21 => :darkred,
    22 => :darkgreen,
    23 => :darkblue,
    24 => :darkcyan,
    25 => :darkmagenta,
    26 => :saddlebrown,
    27 => :darkorange,
    28 => :darkolivegreen,
    29 => :darkslategray,
    30 => :darkorchid,
    31 => :darkgoldenrod,
    32 => :darkseagreen,
    33 => :darkviolet,
    34 => :darkturquoise,
    35 => :sienna,
    36 => :lightcoral,
    37 => :mediumvioletred,
    38 => :firebrick,
    39 => :mediumseagreen,
    40 => :royalblue,
    # Add more colors as needed
)
