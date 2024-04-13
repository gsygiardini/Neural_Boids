using CUDA

function increment_array!(a)
    idx = threadIdx().x + (blockIdx().x - 1) * blockDim().x
    if idx <= length(a)
        a[idx] += 1
    end
    return nothing
end

# Example usage
a = CuArray([1, 2, 3, 4, 5])
@cuda threads=length(a) increment_array!(a)

# Check the modified array
println(a)  # Output: CuArray{Int64, 1}([2, 3, 4, 5, 6])
