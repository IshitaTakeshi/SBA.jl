calc_epsilon(x_true::Array, x_pred::Array) = x_true - x_pred


function calc_epsilon_aj(Aj::SubArray, epsilon_j::SubArray)
    sum(view(Aj, :, :, i)' * view(epsilon_j, :, i) for i in 1:size(Aj, 3))
end


function calc_epsilon_a(indices::Indices, A::Array, epsilon::Array)
    m = n_viewpoints(indices)

    n_pose_params = size(A, 2)
    epsilon_a = Array{Float64}(undef, n_pose_params, m)

    for j in 1:m
        I = points_by_viewpoint(indices, j)

        Aj = view(A, :, :, I)  # [A_ij for i in I]
        epsilon_j = view(epsilon, :, I)  # [epsilon_ij for i in I]

        epsilon_a[:, j] = calc_epsilon_aj(Aj, epsilon_j)
    end

    epsilon_a
end


function calc_epsilon_bi(Bi::SubArray, epsilon_i::SubArray)
    sum(view(Bi, :, :, j)' * view(epsilon_i, :, j) for j in 1:size(Bi, 3))
end


function calc_epsilon_b(indices::Indices, B::Array, epsilon::Array)
    n = n_points(indices)

    n_point_params = size(B, 2)
    epsilon_b = Array{Float64}(undef, n_point_params, n)

    for i in 1:n
        J = viewpoints_by_point(indices, i)

        Bi = view(B, :, :, J)  # [Bij for j in J]
        epsilon_i = view(epsilon, :, J)  # [epsilon_ij for j in J]

        epsilon_b[:, i] = calc_epsilon_bi(Bi, epsilon_i)
    end

    epsilon_b
end


calc_XtX(XS::SubArray) = sum(X' * X for X in eachslice(XS, dims = 3))

calc_Uj(Aj::SubArray) = calc_XtX(Aj)
calc_Vi(Bi::SubArray) = calc_XtX(Bi)


function calc_U(indices::Indices, A::Array)
    n_pose_params = size(A, 2)

    m = n_viewpoints(indices)

    U = Array{Float64}(undef, n_pose_params, n_pose_params, m)

    for j in 1:m
        # Aj = [Aij for i in points_by_viewpoint(j)]
        Aj = view(A, :, :, points_by_viewpoint(indices, j))
        U[:, :, j] = calc_Uj(Aj)
    end

    U
end


function calc_V_inv(indices::Indices, B::Array)
    n_point_params = size(B, 2)

    n = n_points(indices)

    V_inv = Array{Float64}(undef, n_point_params, n_point_params, n)

    for i in 1:n
        # Bi = [Bij for j in viewpoints_by_point(i)]
        Bi = view(B, :, :, viewpoints_by_point(indices, i))
        Vi = calc_Vi(Bi)
        V_inv[:, :, i] = inv(Vi)
    end

    V_inv
end


function calc_W(indices::Indices, A::Array, B::Array)
    @assert size(A, 3) == size(B, 3)

    N = size(A, 3)

    n_pose_params, n_point_params = size(A, 2), size(B, 2)

    W = Array{Float64}(undef, n_pose_params, n_point_params, N)

    for index in 1:N
        W[:, :, index] = view(A, :, :, index)' * view(B, :, :, index)
    end

    W
end


function calc_Y(indices::Indices, W::Array, V_inv::Array)
    Y = copy(W)

    for i in 1:n_points(indices)
        Vi_inv = view(V_inv, :, :, i)

        # Yi = [Yij for j in viewpoints_by_point(i)]
        # note that Yi is still a copy of Wi
        Yi = view(Y, :, :, viewpoints_by_point(indices, i))

        for Yij in eachslice(Yi, dims = 3)
            Yij[:] = Yij * Vi_inv  # overwrite  Yij = Wi * inv(Vi)
        end
    end

    Y
end


function calc_YjWk(Yj::SubArray, Wk::SubArray)
    sum(view(Yj, :, :, i) * view(Wk, :, :, i)' for i in 1:size(Yj, 3))
end



function calc_S(indices::Indices, U::Array, Y::Array, W::Array)
    @assert size(Y) == size(W)

    block(index::Int) = (n_pose_params * (index-1) + 1):(n_pose_params * index)

    n_pose_params = size(U, 1)
    m = n_viewpoints(indices)

    S = Array{Float64}(undef, n_pose_params * m, n_pose_params * m)

    for (j, k) in Base.product(1:m, 1:m)
        ret = shared_point_indices(indices, j, k)

        if isnothing(ret)
            z = zeros(Float64, n_pose_params, n_pose_params)
            S[block(j), block(k)] = z
            continue
        end

        indices_j, indices_k = ret

        Yj = view(Y, :, :, indices_j)
        Wk = view(W, :, :, indices_k)

        S[block(j), block(k)] = -calc_YjWk(Yj, Wk)

        if j == k
            S[block(j), block(j)] += U[:, :, j]
        end
    end

    S
end


function calc_e(indices::Indices, Y::Array, epsilon_a::Array, epsilon_b::Array)
    N = size(Y, 3)
    n_point_params = size(Y, 1)

    Y_epsilon_b = Array{Float64}(undef, n_point_params, N)

    for i in 1:n_points(indices)
        epsilon_bi = epsilon_b[:, i]

        for index in viewpoints_by_point(indices, i)
            Y_epsilon_b[:, index] = view(Y, :, :, index) * epsilon_bi
        end
    end

    e = copy(epsilon_a)
    for j in 1:n_viewpoints(indices)
        sub = view(Y_epsilon_b, :, points_by_viewpoint(indices, j))
        e[:, j] -= sum(sub, dims = 2)
    end

    e
end


function calc_delta_a(S::Array, e::Array)
    delta_a = S \ vec(e)
    n_pose_params, n_viewpoints = size(e)
    reshape(delta_a, n_pose_params, n_viewpoints)
end


function calc_delta_b(indices::Indices, V_inv::Array, W::Array,
                      epsilon_b::Array, delta_a::Array)
    N = size(W, 3)
    n_point_params = size(epsilon_b, 1)

    W_delta_a = Array{Float64}(undef, n_point_params, N)

    # precalculate W_ij * delta_a_j
    for j in 1:n_viewpoints(indices)
        delta_aj = view(delta_a, :, j)
        # Wi = [W_ij for j in J]
        for index in points_by_viewpoint(indices, j)
            W_delta_a[:, index] = view(W, :, :, index)' * delta_aj
        end
    end

    n = n_points(indices)
    delta_b = Array{Float64}(undef, n_point_params, n)
    for i in 1:n
        # sum 'W_ij * delta_a_j' over 'j'
        sub = view(W_delta_a, :, viewpoints_by_point(indices, i))
        delta_b[:, i] = V_inv[:, :, i] * (epsilon_b[:, i] - sum(sub, dims = 2))
    end
    delta_b
end


function sba(indices::Indices, x_true::Array, x_pred::Array, A::Array, B::Array)
    @assert size(A, 3) == size(B, 3)

    U = calc_U(indices, A)
    V_inv = calc_V_inv(indices, B)
    W = calc_W(indices, A, B)
    Y = calc_Y(indices, W, V_inv)
    S = calc_S(indices, U, Y, W)
    epsilon = calc_epsilon(x_true, x_pred)
    epsilon_a = calc_epsilon_a(indices, A, epsilon)
    epsilon_b = calc_epsilon_b(indices, B, epsilon)
    e = calc_e(indices, Y, epsilon_a, epsilon_b)
    delta_a = calc_delta_a(S, e)
    delta_b = calc_delta_b(indices, V_inv, W, epsilon_b, delta_a)

    delta_a, delta_b
end
