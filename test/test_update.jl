function create_jacobian(mask::BitArray, A::Array, B::Array)
    @assert size(A, 3) == size(B, 3)

    N = sum(mask)
    n_points, n_viewpoints = size(mask)
    n_pose_params = size(A, 2)
    n_point_params = size(B, 2)

    n_rows = 2 * N
    n_cols_a = n_pose_params * n_viewpoints
    n_cols_b = n_point_params * n_points
    JA = zeros(Float64, n_rows, n_cols_a)
    JB = zeros(Float64, n_rows, n_cols_b)

    # J' * J should be invertible
    # n_rows(J) >= n_cols(J)
    @assert n_rows >= n_cols_a + n_cols_b

    n_points, n_viewpoints = size(mask, 1), size(mask, 2)

    viewpoint_indices = Array{Int}(undef, N)
    point_indices = Array{Int}(undef, N)

    index = 1
    for i in 1:n_points
        for j in 1:n_viewpoints
            if !mask[i, j]
                continue
            end

            viewpoint_indices[index] = j
            point_indices[index] = i

            row = (index - 1) * 2 + 1

            col = (j-1) * n_pose_params + 1
            JA[row:row+1, col:col+n_pose_params-1] = A[:, :, index]

            col = (i-1) * n_point_params + 1
            JB[row:row+1, col:col+n_point_params-1] = B[:, :, index]

            index += 1
        end
    end

    indices = SBA.Indices(viewpoint_indices, point_indices)
    J = hcat(JA, JB)
    indices, J
end


# there shouldn't be an empty row / column
# (empty means that all row elements / column elements = 0)
# and it seems that at least two '1' elements must be
# found per one row / column
mask = BitArray([
    1 1 1 1 1 1 1 1 1;
    1 0 1 1 1 0 1 1 0;
    1 1 1 1 1 1 0 1 0;
    1 0 0 1 1 1 0 1 1;
    0 0 1 0 0 0 0 0 1
])

N = sum(mask)

x_true = rand(-9:9, 2, N)
x_pred = rand(-9:9, 2, N)
A = randn(Float64, 2, 4, N)
B = randn(Float64, 2, 3, N)
indices, J = create_jacobian(mask, A, B)
delta_a, delta_b = SBA.sba(indices, x_true, x_pred, A, B)

delta = (J' * J) \ (J' * vec(x_true - x_pred))

n_pose_params = size(A, 2)
n_viewpoints = size(mask, 2)
size_A = n_pose_params * n_viewpoints

@assert isapprox(delta[1:size_A], vec(delta_a))
@assert isapprox(delta[size_A+1:end], vec(delta_b))
