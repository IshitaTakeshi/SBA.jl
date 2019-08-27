# x_ij is a keypoint corresponding to the i-th 3D point
# observed from the j-th camera (viewpoint).
# Assume we have a keypoint array X = {x_ij} observed in condition
# n_points = 4, n_viewpoints = 3,
# where x_22 x_31 x_43 are missing.
#
# X            [x_11 x_12 x_13 x_21 x_23 x_32 x_33 x_41 x_42]
# indices          1    2    3    4    5    6    7    8    9
# point index      1    1    1    2    2    3    3    4    4
# viewpoint index  1    2    3    1    3    2    3    1    2

# viewpoints_by_point[1] == [1, 2, 3]  # x_11 x_12 x_13
# viewpoints_by_point[2] == [4, 5]     # x_21 x_23
# viewpoints_by_point[3] == [6, 7]     # x_32 x_33
# viewpoints_by_point[4] == [8, 9]     # x_41 x_42

# points_by_viewpoint[1] == [1, 4, 8]  # x_11 x_21 x_41
# points_by_viewpoint[2] == [2, 6, 9]  # x_12 x_32 x_42
# points_by_viewpoint[3] == [3, 5, 7]  # x_13 x_23 x_33
#
# mask = [
#     1 1 1;  # x_11 x_12 x_13
#     1 0 1;  # x_21      x_23
#     0 1 1;  #      x_32 x_33
#     1 1 0;  # x_41 x_42
# ]


struct Indices
    mask::BitArray
    viewpoints_by_point::Vector{Vector{Int}}
    points_by_viewpoint::Vector{Vector{Int}}
end


n_points(indices::Indices) = length(indices.viewpoints_by_point)
n_viewpoints(indices::Indices) = length(indices.points_by_viewpoint)


is_non_zero(A::BitArray, dim::Int) = all(any(A; dims = dim))


function Indices(point_indices::Array{Int}, viewpoint_indices::Array{Int})
    @assert length(point_indices) == length(viewpoint_indices)

    n_points = maximum(point_indices)
    n_viewpoints = maximum(viewpoint_indices)
    mask = BitArray(undef, n_points, n_viewpoints)

    viewpoints_by_point = [Array{Int}([]) for i in 1:n_points]
    points_by_viewpoint = [Array{Int}([]) for j in 1:n_viewpoints]

    unique_points = Set([])
    unique_viewpoints = Set([])

    for (index, (i, j)) in enumerate(zip(point_indices, viewpoint_indices))
        push!(viewpoints_by_point[i], index)
        push!(points_by_viewpoint[j], index)
        mask[i, j] = 1

        push!(unique_points, i)
        push!(unique_viewpoints, j)
    end

    # they cannot be true if some point / viewpoint indices are missing
    # ex. raises AssertionError if n_viewpoints == 4 and
    # unique_viewpoints == [1, 2, 4]  (3 is missing)
    @assert length(unique_viewpoints) == n_viewpoints
    @assert length(unique_points) == n_points

    Indices(mask, viewpoints_by_point, points_by_viewpoint)
end


function point_indices(indices::Indices, j::Int)
    """
    'point_indices(j)' should return indices of 3D points
    observable from a viewpoint j
    """

    indices.points_by_viewpoint[j]
end


function viewpoint_indices(indices::Indices, i::Int)
    """
    'viewpoint_indices(i)' should return indices of viewpoints
    that can observe a point i
    """

    indices.viewpoints_by_point[i]
end


function shared_point_indices(indices::Indices, j::Int, k::Int)
    """
    j, k: viewpoint indices
    This function returns two indices of points commonly observed from both viewpoints.
    These two indices are corresponding to the first and second view respectively
    """
    mask_j = view(indices.mask, :, j)
    mask_k = view(indices.mask, :, k)

    indices_j = Array{Int}([])
    indices_k = Array{Int}([])

    index_j = 0
    index_k = 0
    for (bit_j, bit_k) in zip(mask_j, mask_k)
        if bit_j == 1
            index_j += 1
        end

        if bit_k == 1
            index_k += 1
        end

        if bit_j & bit_k == 1
            push!(indices_j, index_j)
            push!(indices_k, index_k)
        end
    end

    if length(indices_j) == 0  # (== length(indices_k))
        return nothing  # no shared points found between j and k
    end

    (indices.points_by_viewpoint[j][indices_j],
     indices.points_by_viewpoint[k][indices_k])
end
