clc; clear;

% Parameters
n1 = 800; n2 = 900;
r_vals = [10, 30, 50, 75, 100, 125, 150, 200];
fs_vals = [0.01, 0.02, 0.03, 0.04, 0.06, 0.08, 0.1, 0.15];
num_trials = 15;
lambda = 1 / sqrt(max(n1, n2));

success_matrix = zeros(length(fs_vals), length(r_vals));

% Main loop
for i = 1:length(fs_vals)
    fs = fs_vals(i);
    for j = 1:length(r_vals)
        r = r_vals(j);
        success_count = 0;

        parfor t = 1:num_trials
            [L, S, M] = generate_data(n1, n2, r, fs);
            [L_hat, S_hat] = inexact_alm_rpca(M, lambda);

            rel_err_L = norm(L - L_hat, 'fro') / norm(L, 'fro');
            rel_err_S = norm(S - S_hat, 'fro') / norm(S, 'fro');

            if rel_err_L <= 1e-3 && rel_err_S <= 1e-3
                success_count = success_count + 1;
            end
        end

        success_matrix(i, j) = success_count / num_trials;

        % Save after each (r, fs)
        % save('success_matrix_checkpoint.mat', 'success_matrix');
        fprintf('Completed r = %d, fs = %.2f → Success = %.2f\n', r, fs, success_matrix(i,j));
    end
end

% Plot heatmap
figure;
imagesc(r_vals, fs_vals, success_matrix);
xlabel('Rank r');
ylabel('Sparsity Fraction f_s');
title('Success Probability of RPCA via ALM');
colorbar;
colormap(gray);
set(gca, 'YDir', 'normal');
saveas(gcf, 'plots/success_probability_heatmap.png');

function [L, S, M] = generate_data(n1, n2, r, fs)
    % Low-rank part
    A = randn(n1, r);
    B = randn(n2, r);
    L = A * B';

    % Sparse part
    s = round(fs * n1 * n2);
    S = zeros(n1, n2);
    idx = randperm(n1 * n2, s);
    S(idx) = 3 * randn(1, s);  % sqrt(9) = 3

    % Final matrix
    M = L + S;
end


function [L_hat, S_hat] = inexact_alm_rpca(M, lambda)
    [m, n] = size(M);
    norm_M = norm(M, 'fro');
    L_hat = zeros(m, n);
    S_hat = zeros(m, n);
    Y = M / max(norm(M(:)), norm(M, 1) / lambda);
    
    mu = 1.25 / norm_M;
    mu_bar = mu * 1e7;
    rho = 1.5;
    tol = 1e-5;
    max_iter = 300;

    for iter = 1:max_iter
        % Update low-rank using partial SVD
        [U, S, V] = svds(M - S_hat + (1/mu) * Y, 100); % 100 is conservative
        sigma = diag(S);
        svp = sum(sigma > 1/mu);
        sigma = max(sigma - 1/mu, 0);
        L_hat = U(:, 1:svp) * diag(sigma(1:svp)) * V(:, 1:svp)';

        % Update sparse
        temp = M - L_hat + (1/mu) * Y;
        S_hat = sign(temp) .* max(abs(temp) - lambda / mu, 0);

        % Dual update
        Z = M - L_hat - S_hat;
        Y = Y + mu * Z;
        mu = min(mu * rho, mu_bar);

        % Check stopping criterion
        if norm(Z, 'fro') / norm_M < tol
            break;
        end
    end
end


% Example: Plot one success case
r = 50; fs = 0.03;
[L, S, M] = generate_data(n1, n2, r, fs);
[L_hat, S_hat] = inexact_alm_rpca(M, lambda);

figure;
subplot(2,2,1), imshow(L, []), title('True Low-Rank');
subplot(2,2,2), imshow(S, []), title('True Sparse');
subplot(2,2,3), imshow(L_hat, []), title('Estimated Low-Rank');
subplot(2,2,4), imshow(S_hat, []), title('Estimated Sparse');
sgtitle(sprintf('Successful RPCA: r=%d, fs=%.2f', r, fs));
saveas(gcf, sprintf('plots/RPCA_success_r%d_fs%.2f.png', r, fs));


% Example: Plot one failure case
r = 150; fs = 0.15;
[L, S, M] = generate_data(n1, n2, r, fs);
[L_hat, S_hat] = inexact_alm_rpca(M, lambda);

figure;
subplot(2,2,1), imshow(L, []), title('True Low-Rank');
subplot(2,2,2), imshow(S, []), title('True Sparse');
subplot(2,2,3), imshow(L_hat, []), title('Estimated Low-Rank');
subplot(2,2,4), imshow(S_hat, []), title('Estimated Sparse');
sgtitle(sprintf('Failed RPCA: r=%d, fs=%.2f', r, fs));
saveas(gcf, sprintf('plots/RPCA_fail_r%d_fs%.2f.png', r, fs));
