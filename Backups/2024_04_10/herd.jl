include("./dbscan.jl")
include("./herd_functions.jl")

iniₚₐᵣₜₛ = 10
total_steps = 10

parameters_dict = Dict(
    "maxₚₐᵣₜₛ" => 100,             #Maximum number of particles in the simulation
    "avg_repr" => false,            #Crossover reproduction (true) vs Select best "genes"
    "roulette" => false,            #Kill boids with weighted probabilities
    "weighted_repr" => false,      #Reproduction has a chance to happen depending of fitness difference
    "reflect_walls" => false,      #Confine boids in a closed space(true) vs Periodic boundary condition (false)
    "vision" => false,      #Boids can only perceive other boids at a certain direction
    "potential" => false,          #Boids are penalized for touching each other
    "innertia" => false,           #Boids have rotational innertia(true)
    "fermionic" => false,          #Boids have a hard core(true)
    "box_size" => 1,               #Size of the reservoir where boids are in
    "ε" => 0.5,                    #DB scan parameter
    "min_pts" => 3,                #DB minimum number of points per cluster
    "λ" => 0.0,                 #Rate of mutation
    "β" => 250,                   #Boids vision distance center_of_mass .+= ((exp(- "β" *dist^2.0)/length(Boids)) * fov ) .* boid₂.pos
    "γ" => 0.5,                    #Offspring rate multiplier
    "κ" => 50.0,                   #S curve to avoid boids touching coefficient 1/(1 - exp(-κ(x-x0)))
    "x₀" => 0.07,                  #S curve to avoid boids touching center value 1/(1 - exp(-κ(x-x0)))
    "r" => 0.1,                   #Boid Interaction Radius
    "rr" => 0.1,                  #Boid Reproduction Radius
    "cr" => 0.02,                  #Boid Collision Radius
    "cn" => 0,                     #Boid Child Number
    "neighbors" => Int32[],        #Boid Initial neighbors
    "speed" => 1.0,                #Boid Velocity
    "τ" => 0,                      #Boid Lifetime
)

Boids = create_boids(parameters_dict,iniₚₐᵣₜₛ)

# anim = @animate
for τ in 1:total_steps
# for τ in 1:total_steps
    update_boids!(Boids, parameters_dict, 1e-2)
#     Plots.scatter([b.pos[1] for b in updated_Boids], [b.pos[2] for b in updated_Boids], aspect_ratio=:equal, legend=false)

#     angles = [boid.θ for boid in updated_Boids]
#     positions_x = [boid.pos[1] for boid in updated_Boids]
#     positions_y = [boid.pos[2] for boid in updated_Boids]
#
#     quiver!(positions_x, positions_y, quiver=(0.02*cos.(angles), 0.02*sin.(angles)), color=:blue)
#
#     xlims!(0, 1)
#     ylims!(0, 1)
#
#     if τ % 100 == 0
#         n_boids = length(updated_Boids)
#         println("step: $τ, #boids: $n_boids")
#     end
end

# Save the animation to a GIF file using Plots' gif function
# gif(anim, "./results/boid_animation.mp4", fps = 10)

# println("Saving Models")
# save_models(updated_Boids)
