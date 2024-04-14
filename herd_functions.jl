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

function update_boids!(parameters_dict::Dict, Δt::Float64)

    nₚₐᵣₜₛ = 100#parameters_dict["maxₚₐᵣₜₛ"]
    Boids = create_boids(parameters_dict,nₚₐᵣₜₛ)

    total_steps = parameters_dict["total_steps"]

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
    d_neighbors = Array{Int32, 2}(undef, 8, length(Boids))

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

    d_i = d_i |> cu
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

    W₁ = randn(Float32, nₚₐᵣₜₛ, hiddenₗₑₙ, inputₗₑₙ) |> cu
    W₂ = randn(Float32, nₚₐᵣₜₛ, outₗₑₙ, hiddenₗₑₙ) |> cu

    b₁ = randn(Float32, nₚₐᵣₜₛ, hiddenₗₑₙ) |> cu
    b₂ = randn(Float32, nₚₐᵣₜₛ, outₗₑₙ) |> cu

    cm = zeros(Float32, 2, 2) |> cu

    x = zeros(Float32, inputₗₑₙ) |> cu

    out = zeros(Float32, 3) |> cu

    avg_coord = zeros(Float32, 3) |> cu

    for τ in 1:total_steps
        @cuda threads=length(Boids) update_kernel!(d_i, d_r, d_τ, d_ϕ, d_cn, d_sx, d_rr, d_pos, d_θ, d_speed, W₁, W₂, b₁, b₂, cm, x, out, avg_coord)

        if τ % 100 == 0
            println("step: $τ of $total_steps")
        end
    end

    println("Starting Animation")
    anim_size = 200
    anim = @animate for τ in 1:anim_size
        @cuda threads=length(Boids) update_kernel!(d_i, d_r, d_τ, d_ϕ, d_cn, d_sx, d_rr, d_pos, d_θ, d_speed, W₁, W₂, b₁, b₂, cm, x, out, avg_coord)

        θ = d_θ |> cpu
        pos = d_pos |> cpu

        Plots.scatter(pos[1,:], pos[2,:], aspect_ratio=:equal, legend=false)
        quiver!(pos[1,:], pos[2,:], quiver=(0.02*cos.(θ), 0.02*sin.(θ)), color=:blue)

        xlims!(0, 1)
        ylims!(0, 1)

        if τ % 1 == 0
            println("animation step: $τ of $anim_size")
        end
    end

    gif(anim, "./results/boid_animation.mp4", fps = 10)

end

function update_kernel!(d_i, d_r, d_τ, d_ϕ, d_cn, d_sx, d_rr, d_pos, d_θ, d_speed, W₁, W₂, b₁, b₂, cm, x, out, avg_coord)
    idx = threadIdx().x + (blockIdx().x - 1) * blockDim().x

    box_size=1
    Δt = 1e-2
    β = 1.0
    μ = 1e-5
    avg_θ = 0.0
    d_τ[idx] += 1

    #Get direction from neural network
    ############################################################################################################################
    for jdx in 1:length(d_i)
        if idx != jdx
            dist = 0.0
            @inbounds fov = (1.0 + cos(d_θ[idx] - d_θ[jdx])) #Field of vision
            for kdx in 1:2
                @inbounds dist += (d_pos[kdx,idx] - d_pos[kdx,jdx])^2
            end
            dist = sqrt(dist)

            @inbounds avg_θ += ((exp(-β*dist^2.0)/length(d_i))) * fov * d_θ[jdx]

            for kdx in 1:2
                @inbounds avg_coord[kdx] += ((exp(-β*dist^2)/length(d_i)) * fov) * d_pos[kdx,jdx]
            end
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

    ############################################################################################################################

    #Recalculate particles' coordinates x,y,θ
    ############################################################################################################################
#     if innertia
    @inbounds d_θ[idx] = mod(d_θ[idx] + π*out[size(W₂,3)] * Δt + π, 2.0 * π) - π
#     else
#         d_coords[3,idx] = d_θ[idx]
#     end

    d_pos[1,idx] = d_pos[1,idx] + d_speed[idx] * Δt * cos(d_θ[idx])
    d_pos[2,idx] = d_pos[2,idx] + d_speed[idx] * Δt * sin(d_θ[idx])
    ############################################################################################################################

    #Periodic boundary conditions
    ############################################################################################################################
    d_pos[1,idx] = mod(d_pos[1,idx] + box_size, box_size)
    d_pos[2,idx] = mod(d_pos[2,idx] + box_size, box_size)
    ############################################################################################################################

    #Birth Dynamics
    ############################################################################################################################
    for jdx in 1:length(d_i)
        dist = 0.0
        for kdx in 1:2
            @inbounds dist += (d_pos[kdx,idx] - d_pos[kdx,jdx])^2
        end
        dist = sqrt(dist)

        if dist < d_rr[idx] && d_sx[idx] != d_sx[jdx]
            #Death dynamics - Here I use a accumulated random selector for killing a particle
            ############################################################################################################################
            max_r = 0
            for ldx in 1:length(d_i)
                max_r = d_ϕ[ldx]
            end
            r = max_r * rand()

            kdx = 0
            sum = 0
            while sum < r
                kdx+=1
                sum+=d_ϕ[kdx]
            end

            ############################################################################################################################

            #Assign new boid to the position the previous one was killed
            ############################################################################################################################
            for ldx in 1:size(W₁,2)
                for mdx in size(W₁,3)
                    if rand() < 0.5
                        W₁[kdx,ldx,mdx] = W₁[idx,ldx,mdx] + 2.0 * μ * (rand() - 0.5)
                    else
                        W₁[kdx,ldx,mdx] = W₁[jdx,ldx,mdx] + 2.0 * μ * (rand() - 0.5)
                    end
                end
            end

            for ldx in 1:size(W₂,2)
                for mdx in size(W₂,3)
                    if rand() < 0.5
                        W₂[kdx,ldx,mdx] = W₂[idx,ldx,mdx] + 2.0 * μ * (rand() - 0.5)
                    else
                        W₂[kdx,ldx,mdx] = W₂[jdx,ldx,mdx] + 2.0 * μ * (rand() - 0.5)
                    end
                end
            end

            for ldx in 1:size(b₁,2)
                if rand() < 0.5
                    b₁[kdx,ldx] = b₁[idx,ldx] + 2.0 * μ * (rand() - 0.5)
                else
                    b₁[kdx,ldx] = b₁[jdx,ldx] + 2.0 * μ * (rand() - 0.5)
                end
            end

            for ldx in 1:size(b₂,2)
                if rand() < 0.5
                    b₂[kdx,ldx] = b₂[idx,ldx] + 2.0 * μ * (rand() - 0.5)
                else
                    b₂[kdx,ldx] = b₂[jdx,ldx] + 2.0 * μ * (rand() - 0.5)
                end
            end

            d_cn[idx] += 1
            d_cn[jdx] += 1
        end
    end
    ############################################################################################################################
    #Now we gotta calculate the fitness values
    d_ϕ[idx] = d_cn[idx] / d_τ[idx]
    #I have to add the penalty for when a particle gets close to another

    return
end
