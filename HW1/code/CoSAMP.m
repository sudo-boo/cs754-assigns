tic;

image_size = 32;
algo_name = "CoSaMP";
set(0, 'DefaultFigureVisible', 'off');

k_values = [5, 10, 20, 30, 50, 100, 150, 200];
m_values = 100:100:1000;

plot_k_idxs = [1,5,8];
plot_m_idxs = [5,7];

RMSE = zeros(length(k_values), length(m_values));
ORIG_IMAGES = zeros(image_size, image_size, length(k_values));

% RECONSTRUCTIONS
RECON_IMAGES_P1 = zeros(image_size, image_size, length(m_values), length(plot_k_idxs));
RECON_IMAGES_P2 = zeros(image_size, image_size, length(k_values), length(plot_m_idxs));

% ITERATIONS
for idx_k = 1:length(k_values)
    k = k_values(idx_k);

    coefficients = zeros(image_size^2, 1);
    selected_indices = randperm(image_size^2, k);
    coefficients(selected_indices) = randn(k, 1);
    psi = dctmtx(image_size^2);
    vec_f = psi * coefficients;
    f = reshape(vec_f, image_size, image_size);

    ORIG_IMAGES(:, :, idx_k) = f;
    
    for idx_m = 1:length(m_values)
        m = m_values(idx_m);
        phi = randi([0, 1], m, image_size^2);
        phi(phi == 0) = -1;
        A = phi * psi;
        y = phi * vec_f;

        theta = CoSaMP(A, y, k, image_size);
        f_hat = reshape(psi * theta, image_size, image_size);
        if any(plot_k_idxs == idx_k)
          RECON_IMAGES_P1(:, :, idx_m, find(plot_k_idxs == idx_k)) = f_hat;
        end
        if any(plot_m_idxs == idx_m)
          RECON_IMAGES_P2(:, :, idx_k, find(plot_m_idxs == idx_m)) = f_hat;
        end
        RMSE(idx_k, idx_m) = norm(vec_f - f_hat(:)) / norm(vec_f);
    end
end

% PLOTS
for i = 1:length(k_values)
  z = ORIG_IMAGES(:,:,i);
  imwrite(z, sprintf("images/orig_%s_k=%d.png", algo_name, k_values(i)));
end

for i = 1:length(plot_k_idxs)
  for j = 1:length(m_values)
    z = RECON_IMAGES_P1(:,:,j,i);
    imwrite(z, sprintf("images/recon_p1_%s_k=%d_m=%d.png", algo_name, k_values(plot_k_idxs(i)), m_values(j)));
  end
end

for i = 1:length(plot_m_idxs)
  for j = 1:length(k_values)
    z = RECON_IMAGES_P2(:,:,j,i);
    imwrite(z, sprintf("images/recon_p2_%s_k=%d_m=%d.png", algo_name, k_values(j), m_values(plot_m_idxs(i))));
  end
end

figure;
hold on;
legends = [];
for idx = 1:length(plot_k_idxs)
    k_idx = plot_k_idxs(idx);
    k = k_values(k_idx);
    y_axis = RMSE(k_idx,:);
    x_axis = m_values;
    plot(x_axis, y_axis);
    legends = [legends, sprintf("k=%d", k)];
end
xlabel('m');
ylabel('RMSE');
title(sprintf("RMSE vs m for varying k - %s", algo_name));
legend(legends);
hold off;
saveas(gcf, sprintf("images/%s_k.png", algo_name));

figure;
hold on;
legends = [];
for idx = 1:length(plot_m_idxs)
    m_idx = plot_m_idxs(idx);
    m = m_values(m_idx);
    y_axis = RMSE(:,m_idx);
    x_axis = k_values;
    plot(x_axis, y_axis);
    legends = [legends, sprintf("m=%d", m)];
end
xlabel('k');
ylabel('RMSE');
title(sprintf("RMSE vs k for varying m - %s", algo_name));
legend(legends);
hold off;
saveas(gcf, sprintf("images/%s_m.png", algo_name));

toc;

% Implementing the CoSaMP Algorithm
function res = CoSaMP(A, y, k, image_size)
    [m, n] = size(A);
    norm_A = A ./ sqrt(sum(A.^2, 1));
    r = y;
    res = zeros(n, 1);
    T = [];
    iterations = 0;

    while iterations < m && norm(r)^2 > 1e-6
        % Step 1: Compute proxy
        proxy = A' * r;
        
        % Step 2: Identify large components
        [~, idx] = sort(abs(proxy), 'descend');
        Omega = idx(1:2*k);
        
        % Step 3: Merge support sets
        T = union(T, Omega);
        
        % Step 4: Solve least squares on selected indices
        theta_T = A(:, T) \ y;
        
        % Step 5: Prune to best k coefficients
        [~, prune_idx] = sort(abs(theta_T), 'descend');
        T = T(prune_idx(1:k));
        theta_T = theta_T(prune_idx(1:k));
        
        % Step 6: Update residual
        res(T) = theta_T;
        r = y - A(:, T) * theta_T;
        
        iterations = iterations + 1;
    end
end
