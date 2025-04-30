cvx_quiet(true);

tic;

% constants
N = 500;
M = 300;
VEC_LAMBDA = [0.0001,0.0005,0.001,0.005,0.01,0.05,0.1,0.5,1,2,5,10,15,20,30,50,100];
VEC_THETA_L0 = [5,10,15,20];

% initialize problem params
psi = dct(eye(N));
phi = (1/sqrt(M)) * double(2*randi([0,1],M,N) - 1);

% iterate over lambdas and sparsity values of theta, and use CVX to find theta
for theta_l0_idx = 1:length(VEC_THETA_L0)
  theta_l0 = VEC_THETA_L0(theta_l0_idx);

  % initialize problem params
  theta_orig = 1000*rand(N,1);
  zero_indices = randperm(N, N-theta_l0);
  theta_orig(zero_indices) = 0;
  X = psi*theta_orig;
  sigma = 0.025 * mean(abs(phi*X));
  eta = sigma * randn(M, 1);
  y = phi*X + eta;

  % construct reconstruction and validation sets
  rs_size = floor(0.9*M);
  R_idxs = randperm(M,rs_size);
  V_idxs = setdiff(1:M,R_idxs);

  % construct phi and y for reconstruction and validation sets
  phi_R = phi;
  phi_R(V_idxs,:) = 0;
  phi_V = phi;
  phi_V(R_idxs,:) = 0;
  y_R = y;
  y_R(V_idxs,:) = 0;
  y_V = y;
  y_V(R_idxs,:) = 0;

  min_VE = 1000000;
  min_RMSE = 1000000;
  min_morozov = 1000000;
  lambda_min_VE = VEC_LAMBDA(1);
  lambda_min_RMSE = VEC_LAMBDA(1);
  lambda_min_morozov = VEC_LAMBDA(1);

  VEC_VE = zeros(length(VEC_LAMBDA),1);
  VEC_morozov = zeros(length(VEC_LAMBDA),1);
  VEC_RMSE = zeros(length(VEC_LAMBDA),1);

  for lambda_idx = 1:length(VEC_LAMBDA)
    lambda = VEC_LAMBDA(lambda_idx);
    
    % solve for theta
    cvx_clear;
    cvx_begin
      variable theta(N);
      minimize(sum_square(y_R - phi_R*psi*theta) + lambda*norm(theta,1));
      subject to
        nnz(theta) == theta_l0;
    cvx_end

    % calculate validation error
    VE = sum((y_V-phi_V*psi*theta).^2) / (M-rs_size);
    VEC_VE(lambda_idx) = VE;
    if (VE < min_VE)
      min_VE = VE;
      lambda_min_VE = lambda;
    end

    % calculate morozov error
    morozov_error = abs(norm(y-phi*psi*theta,2)^2 - M*sigma^2);
    VEC_morozov(lambda_idx) = morozov_error;
    if (morozov_error < min_morozov)
      min_morozov = morozov_error;
      lambda_min_morozov = lambda;
    end

    % calculate RMSE
    RMSE = norm(psi*theta-X,2) / norm(X,2);
    VEC_RMSE(lambda_idx) = RMSE;
    if (RMSE < min_RMSE)
      min_RMSE = RMSE;
      lambda_min_RMSE = lambda;
    end
  end

  fprintf("LAMBDA MIN VE: %f, L0 = %f\n",lambda_min_VE,theta_l0);
  fprintf("LAMBDA MIN RMSE: %f, L0 = %f\n",lambda_min_RMSE,theta_l0);
  fprintf("LAMBDA MIN MOZOROV: %f, L0 = %f\n",lambda_min_morozov,theta_l0);

  % plots
  figure,plot(log10(VEC_LAMBDA),VEC_VE);
  xlabel("log(lambda)");
  ylabel("Validation Error");
  title(sprintf("VE vs Lambda (L0 (theta) = %d)",theta_l0));
  set(gcf, 'Position', [100, 100, 500, 500]);
  axis tight;
  saveas(gcf,sprintf("images/VE_vs_LAMBDA_L0=%d.png",theta_l0));

  figure,plot(log10(VEC_LAMBDA),VEC_morozov);
  xlabel("log(lambda)");
  ylabel("Morozov Error");
  title(sprintf("Morozov error vs Lambda (L0 (theta) = %d)",theta_l0));
  set(gcf, 'Position', [100, 100, 500, 500]);
  axis tight;
  saveas(gcf,sprintf("images/morozov_error_vs_LAMBDA_L0=%d.png",theta_l0));

  figure,plot(log10(VEC_LAMBDA),VEC_RMSE);
  xlabel("log(lambda)");
  ylabel("RMSE");
  title(sprintf("RMSE vs Lambda (L0 (theta) = %d)",theta_l0));
  set(gcf, 'Position', [100, 100, 500, 500]);
  axis tight;
  saveas(gcf,sprintf("images/RMSE_vs_LAMBDA_L0=%d.png",theta_l0));
end

toc;