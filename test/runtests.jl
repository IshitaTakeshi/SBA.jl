using Test

import SBA


# x_ij is a keypoint corresponding to the i-th 3D point
# observed from the j-th camera (viewpoint).
# Assume we have a keypoint array X = {x_ij} observed in condition
# n_points = 4, n_viewpoints = 3,
# where x_22 x_31 x_43 are missing.
#
# indices     1    2    3    4    5    6    7    8    9
#   X =   [x_11 x_12 x_13 x_21 x_23 x_32 x_33 x_41 x_42]

# then the corresponding mask should be
mask = BitMatrix([
    1 1 1;  # x_11 x_12 x_13
    1 0 1;  # x_21      x_23
    0 1 1;  #      x_32 x_33
    1 1 0;  # x_41 x_42
])

indices = SBA.Indices(mask)

@test SBA.n_points(indices) == 4
@test SBA.n_viewpoints(indices) == 3

# get array indices of X for x_*1
# where x_*1 are projections of all visible points in the 1st viewpoint
@test SBA.viewpoint_indices(indices, 1) == [1, 4, 8]

# get array indices of X for x_*3
@test SBA.viewpoint_indices(indices, 3) == [3, 5, 7]

# get array indices of X for x_1*
# where x_1* are projections of the 1st 3D point in all observable viewpoints
@test SBA.point_indices(indices, 1) == [1, 2, 3]

# get array indices of X for x_4*
@test SBA.point_indices(indices, 4) == [8, 9]

# [x_12, x_32] and [x_13, x_33] are shared
@test SBA.shared_point_indices(indices, 2, 3) == ([2, 6], [3, 7])

# [x_11, x_21] and [x_13, x_23] are shared
@test SBA.shared_point_indices(indices, 1, 3) == ([1, 4], [3, 5])


# another case
# n_points = 2, n_viewpoints = 3,
# indices     1    2    3
#   X =   [x_13 x_21 x_22]

# then the corresponding mask should be
mask = BitMatrix([
    0 0 1;  #           x_13
    1 1 0;  # x_21 x_22
])

indices = SBA.Indices(mask)

@test SBA.n_points(indices) == 2
@test SBA.n_viewpoints(indices) == 3

# get array indices of X for x_*1
@test SBA.viewpoint_indices(indices, 1) == [2]

# get array indices of X for x_*3
@test SBA.viewpoint_indices(indices, 3) == [1]

# get array indices of X for x_1*
@test SBA.point_indices(indices, 1) == [1]

# get array indices of X for x_2*
@test SBA.point_indices(indices, 2) == [2, 3]

# [x_21] and [x_22] are shared
@test SBA.shared_point_indices(indices, 1, 2) == ([2], [3])

# no points are shared
@test isnothing(SBA.shared_point_indices(indices, 1, 3))


# 2nd row has only zero elements
mask = BitMatrix([
    1 0 1 0;
    0 0 0 0;
    0 1 1 1
])
@test_throws AssertionError SBA.assert_if_zero_row_found(mask)

# last column has only zero elements
mask = BitMatrix([
    1 0 1 0;
    0 1 0 0;
    0 1 1 0
])
@test_throws AssertionError SBA.assert_if_zero_col_found(mask)

# nothing should happeen
mask = BitMatrix([
    1 0 1 0;
    0 1 0 1;
    0 1 1 0
])
SBA.assert_if_zero_col_found(mask)
SBA.assert_if_zero_row_found(mask)
