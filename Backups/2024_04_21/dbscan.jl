 using LinearAlgebra

# Function to calculate Euclidean distance between two points
function euclidean_distance(p₁::Vector, p₂::Vector)
    return sqrt(sum((p₁ .- p₂).^2))
end

# Function to find all points within ε distance from a given point
function region_query(points::Vector{Any}, point_index::Int, ε::Float64)
    neighbors = []
    for i = 1:length(points)
        if euclidean_distance(points[point_index], points[i]) <= ε
            push!(neighbors, i)
        end
    end
    return neighbors
end

# Function to expand a cluster from a seed point
function expand_cluster(points::Vector{Any}, point_index::Int, cluster_id::Int, ε::Float64, minₚₜₛ::Int, clusters::Array{Int})
    neighbors = region_query(points, point_index, ε)
    if length(neighbors) < minₚₜₛ
        clusters[point_index] = -1  # Mark as noise
        return false
    else
        clusters[point_index] = cluster_id
        for i in neighbors
            clusters[i] == 0 && (clusters[i] = cluster_id)
        end
        while !isempty(neighbors)
            current_point = pop!(neighbors)
            current_neighbors = region_query(points, current_point, ε)
            if length(current_neighbors) >= minₚₜₛ
                for i in current_neighbors
                    if clusters[i] <= 0
                        if clusters[i] == 0
                            push!(neighbors, i)
                        end
                        clusters[i] = cluster_id
                    end
                end
            end
        end
        return true
    end
end

# DBSCAN algorithm
function dbscan(points::Vector{Any}, ε::Float64, minₚₜₛ::Int)
    num_points = length(points)
    clusters = zeros(Int, num_points)  # 0: Undefined, -1: Noise, >0: Cluster ID
    current_cluster_id = 0
    for i = 1:num_points
        if clusters[i] == 0
            if expand_cluster(points, i, current_cluster_id + 1, ε, minₚₜₛ, clusters)
                current_cluster_id += 1
            end
        end
    end
    return clusters
end
