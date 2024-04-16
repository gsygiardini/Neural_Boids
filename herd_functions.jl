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

function update_kernel!(d_params, d_τ, d_ϕ, d_cn, d_sx, d_rr, d_pos, d_θ, d_speed, W₁, W₂, b₁, b₂, x, out, avg_coord)
    idx = threadIdx().x + (blockIdx().x - 1) * blockDim().x

    nₚₐᵣₜₛ = Int(d_params[1])
    Δt = d_params[3]
    μ = d_params[4]
    β = d_params[5]
    κ = d_params[6]
    γ = d_params[7]
    x₀ = d_params[8]
    box_size = d_params[9]
    vision = d_params[10]
    innertia = d_params[11]
    avg_repr = d_params[12]
    roulette = d_params[13]
    potential = d_params[14]
    fermionic = d_params[15]
    weighted_repr = d_params[16]
    reflect_walls = d_params[17]
    crossover = d_params[18]

    avg_θ = 0.0
    @inbounds d_τ[idx] += 1

    #Get direction from neural network
    ############################################################################################################################
    for jdx in 1:nₚₐᵣₜₛ
        if idx != jdx
            dist = 0.0
            @inbounds Δx = d_pos[1,idx] - d_pos[1,jdx]
            @inbounds Δy = d_pos[2,idx] - d_pos[2,jdx]

            if vision==true
                # Diference must be limited from -pi/2 to pi/2
                α = atan(Δy,Δx)

                Δθ = d_θ[idx] - α
                if abs(Δθ) > π/2.0
                    Δθ = π/2.0
                end
                @inbounds fov = (1.0 + cos(Δθ)) / 2.0 #Field of vision
#                 CUDA.@cuprint(fov,"\n")
            else
                fov=1.0
            end

            dist = sqrt(Δx^2.0 + Δy^2.0)

            @inbounds avg_θ += ((exp(-β*dist^2.0)/nₚₐᵣₜₛ)) * fov * d_θ[jdx]

            for kdx in 1:2
                @inbounds avg_coord[idx,kdx] += ((exp(-β*dist^2)/nₚₐᵣₜₛ) * fov) * d_pos[kdx,jdx]
            end
        end
    end

    x[idx,1] = avg_coord[idx,1]
    x[idx,2] = avg_coord[idx,2]
    x[idx,3] = avg_θ

    for jdx in 1:size(W₁,3)
        tmp = 0
        for kdx in 1:size(W₁,2)
            @inbounds tmp += W₁[idx,kdx,jdx] * x[kdx]
        end
        out[idx,jdx] = tmp
    end

    for jdx in 1:size(W₂,3)
        tmp = 0
        for kdx in 1:size(W₂,2)
            @inbounds tmp += W₂[idx,kdx,jdx] * tanh(out[idx,kdx] + b₁[idx,kdx])
        end
        out[idx,jdx] = tmp
    end
    @inbounds out[idx,1] = tanh.(out[idx,1] + b₂[idx])
#     CUDA.@cuprint(out[1,1], "\n")

    ############################################################################################################################

    #Recalculate particles' coordinates x,y,θ
    ############################################################################################################################

    #BUG ### TEST TO SEE IF USING THE FIRST INDEX IS CORRECT
    if innertia==true
        @inbounds d_θ[idx] = mod(d_θ[idx] + π*out[idx,1] * Δt + π, 2.0 * π) - π
    else
        @inbounds d_θ[idx] = π*out[idx,1]
    end

    @inbounds d_pos[1,idx] = d_pos[1,idx] + d_speed[idx] * Δt * cos(d_θ[idx])
    @inbounds d_pos[2,idx] = d_pos[2,idx] + d_speed[idx] * Δt * sin(d_θ[idx])
    ############################################################################################################################

    #Periodic boundary conditions
    ############################################################################################################################
#     if !reflect_walls
    @inbounds d_pos[1,idx] = mod(d_pos[1,idx] + box_size, box_size)
    @inbounds d_pos[2,idx] = mod(d_pos[2,idx] + box_size, box_size)
#     else

#     end
    ############################################################################################################################

    #Birth Dynamics
    ############################################################################################################################
    sum_dist = 0.0
    for jdx in 1:nₚₐᵣₜₛ
        dist = 0.0
        for kdx in 1:2
            @inbounds dist += (d_pos[kdx,idx] - d_pos[kdx,jdx])^2
        end
        dist = sqrt(dist)
        sum_dist += dist

        if @inbounds  dist < d_rr[idx] && d_sx[idx] != d_sx[jdx]
            #Death dynamics - Here I use a accumulated random selector for killing a particle
            ############################################################################################################################

            if roulette == true
                #Roulette kill
                max_r = 0
                for ldx in 1:nₚₐᵣₜₛ
                    max_r += d_ϕ[ldx]
                end

                kdx = 0
                while true
                    kdx = 0
                    r = (nₚₐᵣₜₛ-1) * max_r * rand()
                    while r≥0 && kdx < nₚₐᵣₜₛ
                        kdx+=1
                        r -= (max_r - d_ϕ[kdx])
                    end
                    if kdx==0 kdx=1 end

                    if d_τ[kdx] > 1
                        break
                    end
                end
            else
                #Kill Worst
                ############################################################################################################################
                kdx = 1
                min_ϕ = Inf
                for ldx in 1:nₚₐᵣₜₛ
                    if min_ϕ > d_ϕ[ldx] && d_τ[ldx] > 1
                        kdx = ldx
                    end
                end
            end
            ############################################################################################################################

            #Assign new boid to the position the previous one was killed
            ############################################################################################################################
            if crossover == true
                #Crossover
                for ldx in 1:size(W₁,2)
                    for mdx in size(W₁,3)
                        if rand() < 0.5
                            @inbounds W₁[kdx,ldx,mdx] = (1.0 - μ) * W₁[idx,ldx,mdx] + 2.0 * μ * (rand() - 0.5)
                        else
                            W₁[kdx,ldx,mdx] = (1.0 - μ) * W₁[jdx,ldx,mdx] + 2.0 * μ * (rand() - 0.5)
                        end
                    end
                end

                for ldx in 1:size(W₂,2)
                    for mdx in size(W₂,3)
                        if rand() < 0.5
                            @inbounds W₂[kdx,ldx,mdx] = (1.0 - μ) * W₂[idx,ldx,mdx] + 2.0 * μ * (rand() - 0.5)
                        else
                            W₂[kdx,ldx,mdx] = (1.0 - μ) * W₂[jdx,ldx,mdx] + 2.0 * μ * (rand() - 0.5)
                        end
                    end
                end

                for ldx in 1:size(b₁,2)
                    if rand() < 0.5
                        @inbounds b₁[kdx,ldx] = (1.0 - μ) * b₁[idx,ldx] + 2.0 * μ * (rand() - 0.5)
                    else
                        b₁[kdx,ldx] = (1.0 - μ) * b₁[jdx,ldx] + 2.0 * μ * (rand() - 0.5)
                    end
                end

                for ldx in 1:size(b₂,2)
                    if rand() < 0.5
                        @inbounds b₂[kdx,ldx] = (1.0 - μ) * b₂[idx,ldx] + 2.0 * μ * (rand() - 0.5)
                    else
                        b₂[kdx,ldx] = (1.0 - μ) * b₂[jdx,ldx] + 2.0 * μ * (rand() - 0.5)
                    end
                end
            else
                ############################################################################################################################
                #Select Best
                for ldx in 1:size(W₁,2)
                    for mdx in size(W₁,3)
                        if d_ϕ[idx] > d_ϕ[jdx]
                            @inbounds W₁[kdx,ldx,mdx] = (1.0 - μ) * W₁[idx,ldx,mdx] + 2.0 * μ * (rand() - 0.5)
                        else
                            W₁[kdx,ldx,mdx] = (1.0 - μ) * W₁[jdx,ldx,mdx] + 2.0 * μ * (rand() - 0.5)
                        end
                    end
                end

                for ldx in 1:size(W₂,2)
                    for mdx in size(W₂,3)
                        if d_ϕ[idx] > d_ϕ[jdx]
                            @inbounds W₂[kdx,ldx,mdx] = (1.0 - μ) * W₂[idx,ldx,mdx] + 2.0 * μ * (rand() - 0.5)
                        else
                            W₂[kdx,ldx,mdx] = (1.0 - μ) * W₂[jdx,ldx,mdx] + 2.0 * μ * (rand() - 0.5)
                        end
                    end
                end

                for ldx in 1:size(b₁,2)
                    if d_ϕ[idx] > d_ϕ[jdx]
                        @inbounds b₁[kdx,ldx] = (1.0 - μ) * b₁[idx,ldx] + 2.0 * μ * (rand() - 0.5)
                    else
                        b₁[kdx,ldx] = (1.0 - μ) * b₁[jdx,ldx] + 2.0 * μ * (rand() - 0.5)
                    end
                end

                for ldx in 1:size(b₂,2)
                    if d_ϕ[idx] > d_ϕ[jdx]
                        @inbounds b₂[kdx,ldx] = (1.0 - μ) * b₂[idx,ldx] + 2.0 * μ * (rand() - 0.5)
                    else
                        b₂[kdx,ldx] = (1.0 - μ) * b₂[jdx,ldx] + 2.0 * μ * (rand() - 0.5)
                    end
                end
            end
            ###########################################################################################################################

            d_τ[kdx] = 1
            d_cn[kdx] = 0
            @inbounds d_cn[idx] += 1
            d_cn[jdx] += 1
        end
    end
    ############################################################################################################################
    #Now we gotta calculate the fitness values
#     @inbounds d_ϕ[idx] = γ*(d_cn[idx] / d_τ[idx])
    #I have to add the penalty for when a particle gets close to another

    @inbounds d_ϕ[idx] = γ * exp(- sum_dist / nₚₐᵣₜₛ)

    return
end

function order_parameter(nₚₐᵣₜₛ, d_pos, d_θ, d_speed)
    vx = 0
    vy = 0
    sum_speed = 0

    for idx in 1:nₚₐᵣₜₛ
        vx += d_speed[idx] * cos(d_θ[idx])
        vy += d_speed[idx] * sin(d_θ[idx])
        sum_speed = sum_speed + d_speed[idx]
    end
    avg_speed = sum_speed/nₚₐᵣₜₛ

    ψ = sqrt(vx^2 + vy^2)/(nₚₐᵣₜₛ*avg_speed)

    return ψ
end

function calculate_vorticity(nₚₐᵣₜₛ, pos, θₛ)
    vorticity = zeros(nₚₐᵣₜₛ)

    for idx in 1:nₚₐᵣₜₛ
        x, y, θ = pos[1, idx], pos[2, idx], θₛ[idx]
        Δx = 1e-2
        Δy = 1e-2

        vx = cos(θ)
        vy = sin(θ)

        # Approximate partial derivatives using central difference
        dvx_dy = (cos(θ + Δy) - cos(θ - Δy)) / (2Δy)
        dvy_dx = (sin(θ + Δx) - sin(θ - Δx)) / (2Δx)

        # Calculate vorticity
        vorticity[idx] = dvx_dy - dvy_dx
    end

    return vorticity
end

# WRONG
# function vorticity(nₚₐᵣₜₛ, pos, θ)
#     n = length(pos)
#     ω = zeros(nₚₐᵣₜₛ)
#
#     for i in 1:nₚₐᵣₜₛ
#         # Calculate the vorticity at particle i
#         ω_x = sum(θ[j] * (pos[j][2] - pos[i][2]) for j in 1:nₚₐᵣₜₛ if j != i)
#         ω_y = sum(θ[j] * (pos[i][1] - pos[j][1]) for j in 1:nₚₐᵣₜₛ if j != i)
#         ω[i] = ω_x - ω_y
#     end
#
#     return ω
# end

#
# function collisions(Boids::Vector{Boid})
#     #Number of particle collisions
#     counts = 0
#     distances = []
#
#     for i in 1:length(Boids)
#         for j in i+1:length(Boids)
#             distance = norm(Boids[i].pos - Boids[j].pos)
#             push!(distances,distance)
#             if distance < Boids[i].cr
#                 counts+=1
#             end
#         end
#     end
#
#     return counts, mean(distances)
# end
