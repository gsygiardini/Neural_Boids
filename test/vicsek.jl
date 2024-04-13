using CUDA
using ImageIO
using Colors

# Define the Boid struct
struct Boid
    pos_x::Float32
    pos_y::Float32
    vel_x::Float32
    vel_y::Float32
    angle::Float32
end

# Function to initialize the boids
function initialize_boids(n::Int, box_size::Float32)
    boids = Boid[]

    for i in 1:n
        pos_x = rand(Float32)*box_size |> cu
        pos_y = rand(Float32)*box_size |> cu
        vel_x = randn(Float32) |> cu
        vel_y = randn(Float32) |> cu
        angle = atan(vel_y, vel_x) |> cu

        push!(boids, Boid(pos_x, pos_y, vel_x, vel_y, angle))
    end
    return boids
end

# Function to compute the average angle of neighbors
function average_angle(boids::Vector{Boid}, i::Int, r::Float32, box_size::Float32)
    boid = boids[i]
    neighbors = CUDA.CuVector{Int64}()
    for j in 1:length(boids)
        if j != i
            dx = boids[j].pos[1] - boid.pos[1]
            dy = boids[j].pos[2] - boid.pos[2]
            dx = dx - round(dx / box_size) * box_size
            dy = dy - round(dy / box_size) * box_size
            dist = sqrt(dx^2 + dy^2)
            if dist < r
                push!(neighbors, j)
            end
        end
    end

    avg_angle = 0.0f0
    if !isempty(neighbors)
        for j in neighbors
            avg_angle += boids[j].angle
        end
        avg_angle /= length(neighbors)
    end

    return avg_angle
end

function update_kernel_func(boid_pos_x::CuVector{Float32}, boid_pos_y::CuVector{Float32},
                            boid_vel_x::CuVector{Float32}, boid_vel_y::CuVector{Float32},
                            boid_angle::CuVector{Float32},
                            r::Float32, η::Float32, v₀::Float32, box_size::Float32)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if i <= length(boid_pos_x)
        pos_x = boid_pos_x[i]
        pos_y = boid_pos_y[i]
        vel_x = boid_vel_x[i]
        vel_y = boid_vel_y[i]
        angle = boid_angle[i]
        avg_angle = average_angle(boid_pos_x, boid_pos_y, boid_vel_x, boid_vel_y, boid_angle, i, r, box_size)
        angle = angle + η * (avg_angle - angle)
        vel_x = CUDA.cos(angle) * v₀
        vel_y = CUDA.sin(angle) * v₀
        pos_x += vel_x
        pos_y += vel_y
        pos_x = mod(pos_x, box_size)
        pos_y = mod(pos_y, box_size)
        boid_pos_x[i] = pos_x
        boid_pos_y[i] = pos_y
        boid_vel_x[i] = vel_x
        boid_vel_y[i] = vel_y
        boid_angle[i] = angle
    end
    return nothing
end


#Function to update the boids
function update_boids!(boid_pos_x::CuVector{Float32}, boid_pos_y::CuVector{Float32},
                       boid_vel_x::CuVector{Float32}, boid_vel_y::CuVector{Float32},
                       boid_angle::CuVector{Float32},
                       r::Float32, η::Float32, v₀::Float32, box_size::Float32)
    n = length(boid_pos_x)
    CUDA.@sync begin
        @cuda threads=n update_kernel_func(boid_pos_x, boid_pos_y, boid_vel_x, boid_vel_y, boid_angle, r, η, v₀, box_size)
    end
    nothing
end


# Function to run the simulation and create a GIF animation
function run_simulation_with_animation(n::Int, steps::Int, r::Float32, η::Float32, v₀::Float32, box_size::Float32, filename::String)
    boids = initialize_boids(n, box_size)
    frames = Vector{Matrix{RGB}}(undef, steps)
#
    boid_pos_x = CUDA.CuVector(map(boid -> boid.pos_x, boids))
    boid_pos_y = CUDA.CuVector(map(boid -> boid.pos_y, boids))
    boid_vel_x = CUDA.CuVector(map(boid -> boid.vel_x, boids))
    boid_vel_y = CUDA.CuVector(map(boid -> boid.vel_y, boids))
    boid_angle = CUDA.CuVector(map(boid -> boid.angle, boids))
#
    for step in 1:steps
        update_boids!(boid_pos_x, boid_pos_y, boid_vel_x, boid_vel_y, boid_angle, r, η, v₀, box_size)

        # Capture the positions of the boids for this time step
        positions = [boid_pos[i] for i in 1:length(boids)]

        # Create a frame from the positions
        frame = create_frame(positions, box_size)
        frames[step] = frame
    end
#
#     # Save the frames as a GIF animation
#     save("animation.gif", frames)
end

# # Helper function to create a frame from the boid positions
# function create_frame(positions::Vector{CuVector{Float32}}, box_size::Float32)
#     frame = fill(RGB(0.0, 0.0, 0.0), (round(Int, box_size), round(Int, box_size)))
#
#     for pos in positions
#         x = round(Int, pos[1])
#         y = round(Int, pos[2])
#         frame[y, x] = RGB(1.0, 1.0, 1.0)  # Set the pixel to white
#     end
#
#     return frame
# end

# Example usage
n = 1000
steps = 1000
r = 1.0f0
η = 0.5f0
v₀ = 0.03f0
box_size = 10.0f0

run_simulation_with_animation(n, steps, r, η, v₀, box_size, "animation.gif")
