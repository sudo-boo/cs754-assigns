tic;

addpath("../MMread");

PATCH_SIZE = 8;
H_MAX = 120;
W_MAX = 240;

solve('../cars.avi',"../images/cars_coded_","../images/cars_",3,PATCH_SIZE,H_MAX,W_MAX);
solve('../cars.avi',"../images/cars_coded_","../images/cars_",5,PATCH_SIZE,H_MAX,W_MAX);
solve('../cars.avi',"../images/cars_coded_","../images/cars_",7,PATCH_SIZE,H_MAX,W_MAX);
solve('../flame.avi',"../images/flame_coded_","../images/flame_",5,PATCH_SIZE,1000,1000);

toc;

function solve(original_path, coded_path, recon_path, T, PATCH_SIZE, H_MAX, W_MAX)
  A = mmread(original_path);
  [H,W] = size(rgb2gray(A.frames(1).cdata));

  X = zeros(H,W,T);
  for i=1:T
    X(:,:,i) = double(rgb2gray(A.frames(i).cdata));
  end

  H = min(H,H_MAX);
  W = min(W,W_MAX);
  X = X(1:H,1:W,:);

  C = randi([0,1],H,W,T);
  I = sum(X.*C,3) + 2*randn(H,W);
  figure, imshow(I, []);
  saveas(gcf, coded_path + sprintf("%d.png",T));

  I_recon = zeros(H,W,T);
  I_counter = zeros(H,W,T);
  psi = kron(kron(dctmtx(8),dctmtx(8)),dctmtx(T));

  for i = 1:(H-PATCH_SIZE+1)
    disp(i);
    for j = 1:(W-PATCH_SIZE+1)
      phi = zeros(PATCH_SIZE*PATCH_SIZE, PATCH_SIZE*PATCH_SIZE*T);
      for t = 1:T
        patch = C(i:(i+PATCH_SIZE-1),j:(j+PATCH_SIZE-1),t);
        phi(:,(1+(t-1)*(PATCH_SIZE*PATCH_SIZE)):(t*(PATCH_SIZE*PATCH_SIZE))) = diag(patch(:));
      end

      y = I(i:(i+PATCH_SIZE-1),j:(j+PATCH_SIZE-1));
      y = y(:);

      x_recon = psi * OMP(phi*psi, y);
      patch_recon = reshape(x_recon,PATCH_SIZE,PATCH_SIZE,T);
      
      I_recon(i:(i+PATCH_SIZE-1),j:(j+PATCH_SIZE-1),:) = I_recon(i:(i+PATCH_SIZE-1),j:(j+PATCH_SIZE-1),:) + patch_recon;
      I_counter(i:(i+PATCH_SIZE-1),j:(j+PATCH_SIZE-1),:) = I_counter(i:(i+PATCH_SIZE-1),j:(j+PATCH_SIZE-1),:) + 1;
    end
  end

  I_recon = I_recon ./ I_counter;

  for t = 1:T
    figure, imshow(I_recon(:,:,t), []);
    saveas(gcf, recon_path + sprintf("%d_%d.png",t,T));
  end

  rmse = sqrt(mean((X-I_recon).^2,"all"));
  disp(sprintf("RMSE: %.4f",rmse));
end

function res = OMP(A, y)
  [m, n] = size(A);
  norm_A = A ./ sqrt(sum(A.^2, 1));
  r = y;
  T = [];
  iterations = 0;

  while iterations < m && norm(r)^2 > 1e-6
      [~, j] = max(abs(norm_A' * r));
      if ismember(j,T)
        break;
      end
      T = [T, j];
      theta = pinv(A(:,T))*y;
      r = y - A(:, T) * theta;
      iterations = iterations + 1;
  end

  res = zeros(n, 1);
  res(T) = theta;
end