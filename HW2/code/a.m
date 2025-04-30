patch_size = 8;
lambda = 1.0;
frame_size = 256;

tic;

solve_a("barbara256.png", patch_size, frame_size, lambda, "images/barbara256_recon_a.png");

toc;

function solve_a(path, patch_size, frame_size, lambda, save_path)
  phi = eye(patch_size*patch_size);
  psi = kron(dctmtx(patch_size)',dctmtx(patch_size)');
  A = phi * psi;

  X = double(imread(path));
  X = X(1:frame_size, 1:frame_size);

  X_recon = zeros(size(X));
  X_noisy = X + 2*randn(size(X));
  X_patches = im2col(X_noisy, [patch_size, patch_size]);

  patch_recon_count = zeros(size(X_recon));
  [~,num_patches] = size(X_patches);
  Y = phi * X_patches;

  X_patches_recon = psi * ista(Y,A,lambda);

  for patch_idx = 1:num_patches
    patch = X_patches_recon(:,patch_idx);
    patch = reshape(patch, [patch_size, patch_size]);
    
    row = mod(patch_idx-1, size(X,1)-patch_size+1)+1;
    col = floor((patch_idx-1) / (size(X,1)-patch_size+1))+1;

    X_recon(row:(row+patch_size-1), col:(col+patch_size-1)) = X_recon(row:(row+patch_size-1), col:(col+patch_size-1)) + patch;
    patch_recon_count(row:(row+patch_size-1), col:(col+patch_size-1)) = patch_recon_count(row:(row+patch_size-1), col:(col+patch_size-1)) + 1;
  end

  X_recon = X_recon ./ patch_recon_count;

  RMSE = norm(X(:) - X_recon(:)) / norm(X(:));
  disp(sprintf("RECONSTRUCTED_RMSE: %.4f", RMSE));

  RMSE = norm(X(:) - X_noisy(:)) / norm(X(:));
  disp(sprintf("NOISY_RMSE: %.4f", RMSE));

  % X_recon = (X_recon - min(X_recon, [], "all")) / (max(X_recon, [], "all") - min(X_recon, [], "all"));
  X_recon = X_recon / 255;
  figure,imshow(X_recon, []);
  saveas(gcf, save_path);
end

function theta = ista(y, A, lambda)
  theta = zeros(size(A, 2), size(y, 2));
  alpha = eigs(A' * A, 1);
  
  for iter = 1:50
    theta = wthresh(theta + (A'/alpha)*(y-A*theta), "s", lambda/(2*alpha));
  end
end

