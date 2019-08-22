# The current implementation is inefficient: consumes a lot of memory space.
# TODO Use sussinct bit vector to reduce the memory consumption
# whilist keeping the speed

struct Indices
    mask::BitMatrix
    indices::Array{Int}
end


n_points(indices::Indices) = size(indices.mask, 1)
n_viewpoints(indices::Indices) = size(indices.mask, 2)


# We don't accept masks that have zero row / col
#
# mask = [1 0 1 0;
#         0 0 0 0;
#         0 1 1 1]
# -> 2nd row has only zero elements. Not accepted.
#
# mask = [1 0 1 0;
#         0 1 0 0;
#         0 1 1 0]
# -> Last column has only zero elements. Not accepted

is_non_zero(A::BitArray, dim::Int) = all(any(A; dims = dim))

assert_if_zero_col_found(A::BitMatrix) = @assert is_non_zero(A, 1)
assert_if_zero_row_found(A::BitMatrix) = @assert is_non_zero(A, 2)


function Indices(mask::BitMatrix)
    assert_if_zero_row_found(mask)
    assert_if_zero_col_found(mask)

    indices = Array{Int}(undef, size(mask))

    n_points, n_viewpoints = size(mask)

    s = 0
    for i in 1:n_points
        for j in 1:n_viewpoints
            s = s + mask[i, j]
            indices[i, j] = s
        end
    end

    Indices(mask, indices)
end


function viewpoint_indices(indices::Indices, j::Int)
    indices.indices[indices.mask[:, j], j]
end

function point_indices(indices::Indices, i::Int)
    indices.indices[i, indices.mask[i, :]]
end


function shared_point_indices(indices::Indices, j::Int, k::Int)
    """
    j, k: viewpoint indices
    This function returns two indices of points commonly observed from both viwpoints.
    These two indices are corresponding to the first and second view respectively
    """

    # element wise multiplication for calculating 'and'
    mask = view(indices.mask, :, j) .* view(indices.mask, :, k)

    # return Nothing if mask is a zero array: no points are shared
    if !any(mask)
        return Nothing
    end

    indices.indices[mask, j], indices.indices[mask, k]
end
