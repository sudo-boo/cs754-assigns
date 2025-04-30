clc; clear all; close all;
rng(42);

Ns = [50, 100, 500, 1000, 2000, 5000, 10000];
img = double(imread("cryoem.png")) / 255.0;

W_exp_denom = 0.01;

for N = Ns
    tic;
    cryoem_test(img, N, W_exp_denom, true, true);
    toc;
end

function cryoem_test(image, num_projections, W_exp_denom, generate_plots, compare_images)
    
    angles = sort(359 * rand(num_projections, 1));
    projections = radon(image, angles);  % size: [proj_size, num_projs]

    estimated_angles = estimate_angles(projections, W_exp_denom);
    reconstructed_image = iradon(projections, estimated_angles);
    
    [H, W] = size(image);
    cropped_reconstructed_image = reconstructed_image(1:H, 1:W);
    assert(isequal(size(cropped_reconstructed_image), size(image)), "cropping failed");

    [best_rmse, alignment_angle] = compute_rmse(image, cropped_reconstructed_image);
    final_reconstruction = imrotate(cropped_reconstructed_image, alignment_angle);

    if compare_images
        plot_images(image, final_reconstruction, best_rmse, num_projections);
    end

    if generate_plots
        plot_angles(angles, estimated_angles, num_projections);
    end
end

function [estimated_angles] = estimate_angles(projections, W_exp_denom)
    [~, num_projections] = size(projections);

    dist_matrix = squareform(pdist(projections', 'euclidean')).^2;
    W = exp(-dist_matrix / W_exp_denom);
    D = diag(sum(W, 2));  % Row sum

    L = D - W;
    A = D \ L;

    [eig_vecs, eig_vals] = eig(A);
    [~, min_eig_ids] = mink(diag(eig_vals), 3);  

    Y = eig_vecs(:, min_eig_ids(2:3));
    
    psi = atan(Y(:, 1) ./ (Y(:, 2) + eps));
       

    [~, sorted_projection_ids] = sort(psi);
    estimated_angles = sorted_projection_ids * (360 / num_projections);
end

function [best_rmse, alignment_angle] = compute_rmse(original, recon)
    best_rmse = inf;
    best_angle = 0;
    
    for theta = 0:0.5:359.5
         rotated_recon = imrotate(recon, theta, 'bilinear', 'crop');
         error = norm(original(:) - rotated_recon(:)) / norm(original(:));
         if error < best_rmse
             best_rmse = error;
             best_angle = theta;
         end
    end

    alignment_angle = best_angle;
end

%% Plotters
function plot_angles(angles, estimated_angles, N)
    fig = figure;
    fig.Position(3:4) = [1000, 400];

    plot(angles, estimated_angles, 'bo-', 'LineWidth', 1, 'MarkerSize', 2); 
    
    hold on;
    plot(angles, angles, 'r--', 'LineWidth', 2); 
    hold off;
    
    xlabel('Actual Angles');
    ylabel('Estimated Angles');
    
    legend('Estimated Angles', 'Actual Angles', 'Location', 'Best');
    title('Comparison of Angles and Estimated Angles');
    
    grid on;
    filename = sprintf('plot_N=%d.png', N);
    exportgraphics(fig, filename, 'Resolution', 300);
end

function plot_images(original, reconstructed, best_rmse, N)
    fig = figure;
    fig.Position(3:4) = [1000, 400];
    
    tiledlayout(1, 3, 'TileSpacing', 'compact', 'Padding', 'compact');
    
    nexttile;
    imshow(uint8(original * 255), []);
    title("Original Image", 'FontSize', 14, 'FontWeight', 'bold');
    
    nexttile;
    imshow(uint8(reconstructed * 255), []);
    title_text = sprintf("Reconstructed Image, RMSE = %.4f", best_rmse);
    title(title_text, 'FontSize', 14, 'FontWeight', 'bold');

    filename = sprintf('reconstruction_N=%d.png', N);
    exportgraphics(fig, filename, 'Resolution', 300);
end

