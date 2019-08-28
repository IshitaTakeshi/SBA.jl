@test SBA.calc_epsilon([1 2 3; 3 1 2], [4 3 2; 5 0 1]) == [-3 -1 1; -2 1 1]

A = Array{Float64}(undef, 2, 4, 4)

A[:, :, 1] = [ 1  3  0 -1;
               2 -1 -1  4]
A[:, :, 2] = [ 3  0  4  1;
               2  1 -1 -3]
A[:, :, 3] = [-1  4  2  0;
               3  1  3  3]
A[:, :, 4] = [ 1  0  3  2;
               0  2  1  1]

AtA = Array{Float64}(undef, 4, 4, 4)
for index in 1:size(A, 3)
    AtA[:, :, index] = A[:, :, index]' * A[:, :, index]
end

B = Array{Float64}(undef, 2, 3, 4)

B[:, :, 1] = [ 2  3  1;
               1  0  2]
B[:, :, 2] = [ 1  0  2;
               1  1  3]
B[:, :, 3] = [ 1  2  3;
               1  0  1]
B[:, :, 4] = [ 0  0  5;
               4  0  3]

BtB = Array{Float64}(undef, 3, 3, 4)
for index in 1:size(B, 3)
    BtB[:, :, index] = B[:, :, index]' * B[:, :, index]
end

# 2 points, 3 viewpoints
# i = 1:2, j = 1:3
# N = 4 (number of visible points)
# X    = [x_11 x_12 x_22 x_23]
# mask = [   1    1    0;
#            0    1    1]

viewpoint_indices = [1, 2, 2, 3]
point_indices = [1, 1, 2, 2]

indices = SBA.Indices(viewpoint_indices, point_indices)

U = SBA.calc_U(indices, A)
@test size(U) == (4, 4, 3)  # (n_pose_paramas, n_pose_paramas, n_viewpoints)
@test U[:, :, 1] == AtA[:, :, 1]
@test U[:, :, 2] == AtA[:, :, 2] + AtA[:, :, 3]
@test U[:, :, 3] == AtA[:, :, 4]

V_inv = SBA.calc_V_inv(indices, B)
@test size(V_inv) == (3, 3, 2)  # (n_point_params, n_point_params, n_points)
@test V_inv[:, :, 1] == inv(BtB[:, :, 1] + BtB[:, :, 2])
@test V_inv[:, :, 2] == inv(BtB[:, :, 3] + BtB[:, :, 4])

W = SBA.calc_W(indices, A, B)
@test size(W) == (4, 3, 4)  # (n_pose_params, n_point_params, N)
@test W[:, :, 1] == A[:, :, 1]' * B[:, :, 1]
@test W[:, :, 2] == A[:, :, 2]' * B[:, :, 2]
@test W[:, :, 3] == A[:, :, 3]' * B[:, :, 3]
@test W[:, :, 4] == A[:, :, 4]' * B[:, :, 4]

epsilon = [ 1  2 -3  1;
            3 -1  3  0]

epsilon_a = SBA.calc_epsilon_a(indices, A, epsilon)
@test size(epsilon_a) == (4, 3)  # (n_pose_params, n_viewpoints)
@test epsilon_a[:, 1] == A[:, :, 1]' * epsilon[:, 1]
@test epsilon_a[:, 2] == A[:, :, 2]' * epsilon[:, 2] +
                         A[:, :, 3]' * epsilon[:, 3]
@test epsilon_a[:, 3] == A[:, :, 4]' * epsilon[:, 4]

epsilon_b = SBA.calc_epsilon_b(indices, B, epsilon)
@test size(epsilon_b) == (3, 2)  # (n_point_params, n_points)
@test epsilon_b[:, 1] == B[:, :, 1]' * epsilon[:, 1] +
                         B[:, :, 2]' * epsilon[:, 2]
@test epsilon_b[:, 2] == B[:, :, 3]' * epsilon[:, 3] +
                         B[:, :, 4]' * epsilon[:, 4]

Y = SBA.calc_Y(indices, W, V_inv)
@test size(Y) == (4, 3, 4)  # (n_pose_params, n_point_params, N)
@test Y[:, :, 1] == W[:, :, 1] * V_inv[:, :, 1]  # (i, j) = (1, 1)
@test Y[:, :, 2] == W[:, :, 2] * V_inv[:, :, 1]  # (i, j) = (1, 2)
@test Y[:, :, 3] == W[:, :, 3] * V_inv[:, :, 2]  # (i, j) = (2, 2)
@test Y[:, :, 4] == W[:, :, 4] * V_inv[:, :, 2]  # (i, j) = (2, 3)

e = SBA.calc_e(indices, Y, epsilon_a, epsilon_b)
@test size(e) == (4, 3)  # (n_pose_params, n_viewpoints)
@test e[:, 1] ==  - Y[:, :, 1] * epsilon_b[:, 1] + epsilon_a[:, 1]
@test e[:, 2] == (- Y[:, :, 2] * epsilon_b[:, 1] - Y[:, :, 3] * epsilon_b[:, 2]
                  + epsilon_a[:, 2])
@test e[:, 3] ==  - Y[:, :, 4] * epsilon_b[:, 2] + epsilon_a[:, 3]

S = SBA.calc_S(indices, U, Y, W)
# (n_pose_params * n_viewpoints, n_pose_params * n_viewpoints)
@test size(S) == (4 * 3, 4 * 3)
# (j, k) == (1, 1)
@test S[1: 4, 1: 4] ==  - Y[:, :, 1] * W[:, :, 1]' + U[:, :, 1]
# (j, k) == (1, 2)
@test S[1: 4, 5: 8] ==  - Y[:, :, 1] * W[:, :, 2]'
# (j, k) == (1, 3)
@test S[1: 4, 9:12] == zeros(Float64, 4, 4)
# (j, k) == (2, 1)
@test S[5: 8, 1: 4] ==  - Y[:, :, 2] * W[:, :, 1]'
# (j, k) == (2, 2)
@test S[5: 8, 5: 8] == (- Y[:, :, 2] * W[:, :, 2]' - Y[:, :, 3] * W[:, :, 3]'
                        + U[:, :, 2])
# (j, k) == (2, 3)
@test S[5: 8, 9:12] ==  - Y[:, :, 3] * W[:, :, 4]'
# (j, k) == (3, 1)
@test S[9:12, 1: 4] == zeros(Float64, 4, 4)
# (j, k) == (3, 2)
@test S[9:12, 5: 8] ==  - Y[:, :, 4] * W[:, :, 3]'
# (j, k) == (3, 3)
@test S[9:12, 9:12] ==  - Y[:, :, 4] * W[:, :, 4]' + U[:, :, 3]


delta_a = SBA.calc_delta_a(S, e)
@test size(delta_a) == (4, 3)  # (n_pose_params, n_viewpoints)
@test isapprox(S * vec(delta_a), vec(e))

delta_b = SBA.calc_delta_b(indices, V_inv, W, epsilon_b, delta_a)
@test size(delta_b) == (3, 2)  # (n_point_params, n_points)

# size(x_true) = (2, N)
x_true = Float64[1 -1 -1  3;
                 0  2  1 -1]

x_pred = Float64[ 0 1 -2 -1;
                 -1 2  0  1]

# mask = [1 1 0;
#         0 1 1]
viewpoint_indices = [1, 2, 2, 3]
point_indices = [1, 1, 2, 2]
indices = SBA.Indices(viewpoint_indices, point_indices)
delta_a, delta_b = SBA.sba(indices, x_true, x_pred, A, B)
@test isapprox(delta_b[:, 1],
               V_inv[:, :, 1] * (epsilon_b[:, 1]
                              - (W[:, :, 1]' * delta_a[:, 1] +
                                 W[:, :, 2]' * delta_a[:, 2])))
@test isapprox(delta_b[:, 2],
               V_inv[:, :, 2] * (epsilon_b[:, 2]
                              - (W[:, :, 3]' * delta_a[:, 2] +
                                 W[:, :, 4]' * delta_a[:, 3])))
