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
@test isapprox(delta_b[:, 1],
               V_inv[:, :, 1] * (epsilon_b[:, 1]
                              - (W[:, :, 1]' * delta_a[:, 1] +
                                 W[:, :, 2]' * delta_a[:, 2])))
@test isapprox(delta_b[:, 2],
               V_inv[:, :, 2] * (epsilon_b[:, 2]
                              - (W[:, :, 3]' * delta_a[:, 2] +
                                 W[:, :, 4]' * delta_a[:, 3])))

# size(J) == (2 * n_points * n_viewpoints,
#             n_viewpoints * n_pose_params + n_points * n_point_params)

size_A = 12
size_B = 6

J = zeros(Float64, 2 * 2 * 3, size_A + size_B)
J[ 1: 2,  1: 4] = A[:, :, 1]
J[ 3: 4,  5: 8] = A[:, :, 2]
J[ 9:10,  5: 8] = A[:, :, 3]
J[11:12,  9:12] = A[:, :, 4]
J[ 1: 2, 13:15] = B[:, :, 1]
J[ 3: 4, 13:15] = B[:, :, 2]
J[ 9:10, 16:18] = B[:, :, 3]
J[11:12, 16:18] = B[:, :, 4]

# X    = [x_11 x_12 x_22 x_23]
# mask = [   1    1    0;
#            0    1    1]
x_true = Float64[ 1 -1 -1  3;
                  0  2  1 -8]

x_pred = Float64[ 0  1 -2 -1;
                 -1  2  9  7]

epsilon = zeros(Float64, 2, 6)
epsilon[:, 1] = (x_true - x_pred)[:, 1]  # x_11
epsilon[:, 2] = (x_true - x_pred)[:, 2]  # x_12
epsilon[:, 5] = (x_true - x_pred)[:, 3]  # x_22
epsilon[:, 6] = (x_true - x_pred)[:, 4]  # x_23

# zero elements are suppressed
# therefore size(epsilon_pred) == (2, 4)
epsilon_pred = SBA.calc_epsilon(x_true, x_pred)

U = (J' * J)[1:size_A, 1:size_A]
U_pred = SBA.calc_U(indices, A)
@test isapprox(U_pred[:, :, 1], U[1:4, 1:4])
@test isapprox(U_pred[:, :, 2], U[5:8, 5:8])

V_inv = inv((J' * J)[size_A+1:size_A+size_B, size_A+1:size_A+size_B])
V_inv_pred = SBA.calc_V_inv(indices, B)
@test isapprox(V_inv_pred[:, :, 1], V_inv[1:3, 1:3])
@test isapprox(V_inv_pred[:, :, 2], V_inv[4:6, 4:6])

W = (J' * J)[1:size_A, size_A+1:size_A+size_B]
W_pred = SBA.calc_W(indices, A, B)
@test isapprox(W_pred[:, :, 1], W[1: 4, 1:3])  # W_11
@test isapprox(W_pred[:, :, 2], W[5: 8, 1:3])  # W_12
@test isapprox(W_pred[:, :, 3], W[5: 8, 4:6])  # W_22
@test isapprox(W_pred[:, :, 4], W[9:12, 4:6])  # W_23

S = U - W * V_inv * W'

Y_pred = SBA.calc_Y(indices, W_pred, V_inv_pred)
S_pred = SBA.calc_S(indices, U_pred, Y_pred, W_pred)
@test isapprox(S, S_pred)

epsilon_a = J[:, 1:size_A]' * vec(epsilon)
epsilon_b = J[:, size_A+1:size_A+size_B]' * vec(epsilon)

epsilon_a_pred = SBA.calc_epsilon_a(indices, A, epsilon_pred)
@test isapprox(vec(epsilon_a_pred), epsilon_a)

epsilon_b_pred = SBA.calc_epsilon_b(indices, B, epsilon_pred)
@test isapprox(vec(epsilon_b_pred), epsilon_b)

e = epsilon_a - W * V_inv * epsilon_b
e_pred = SBA.calc_e(indices, Y_pred, epsilon_a_pred, epsilon_b_pred)
@test isapprox(e, vec(e_pred))
