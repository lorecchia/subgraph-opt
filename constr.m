clear all;
close all;
% cvx_solver sedumi

%% data generation, generate n1xn2 lattice, 4-connected, small example
n1 = 5;
n2 = 5;
n = n1*n2;
K = 4;
xx_line = zeros(n,1);
xx_line([7, 9, 10, 15]) = 1;  % 4 nodes with value 1, 1 node disconnected
xx = reshape(xx_line, n1, n2);

% observations with no noise
yy = xx; % + 0.01*randn(size(xx));

%% data generation, generate n1xn2 lattice, 4-connected, larger example
% (slow and takes a lot of memory)
n1 = 8;
n2 = 8;
n = n1*n2;
K = 9;

xx = zeros(n1, n2);
xx(3:4, 2:4) = 1;  % larger connected component
xx(5, 6:8) = 1;  % smaller connected component

% observations with some noise
yy = xx + 0.01*randn(size(xx));

%% compute adjacency matrix, Laplacian matrix
% unweighted adjacency matrix
Adj = zeros(n, n);
for i=1:n1
    for j=1:n2
        if j<n2
            Adj((i-1)*n2+j, (i-1)*n2+j+1) = 1;
            Adj((i-1)*n2+j+1, (i-1)*n2+j) = 1;
        end
        if i<n1
            Adj((i-1)*n2+j, i*n2+j) = 1;
            Adj(i*n2+j, (i-1)*n2+j) = 1;
        end
    end
end
% unnormalized graph Laplacian
d = sum(Adj,2);
D = diag(d);
L = D-Adj;

% plot using imagesc to display as matrix
figure, imagesc(xx), title('Original');

%% cvx primal opt
gamma = 0.1;  % connectivity parameter
p = 9;  % anchor

cvx_begin
    variable M(n,n) symmetric  % main variable
    expression L_M(n,n)  % laplacian of M
    expression A_M(n,n)  % A.M
    expression L_A_M(n,n)  % laplacian of A.M

    L_M = diag(sum(M,2)) - M;
    A_M = Adj.*M;
    L_A_M = diag(sum(A_M,2)) - A_M;

    maximize( trace( diag(yy(:)) .* M ) )
    subject to
        M == semidefinite(n)
        M >= 0
        trace( M ) <= 1
        sum(sum(M)) <= K
        L_A_M - gamma*L_M == semidefinite(n)

        % constraint so that not all is concentrated on one diagonal elt if diags of yy are
        % not exactly equal (e.g. noisy y), forces the solution to have exactly K non-zero diags
        diag(M) <= 1/K

        % constaints from NIPS paper
%         M <= 1
%         M(p,p) == 1
%         for i = 1:n
%             M(i,i) <= M(p,i)
%             M(i,:) <= M(i,i)
%         end
cvx_end

% show result using imagesc, no thresholding
figure, imagesc(reshape(diag(M), n1, n2)), colorbar, title('Estimated');

%% project and display

S = diag(M) > 1e-3;
Msub = M(find(S), find(S));  % non-zero submatrix of M

% lattice coordinates of indices in M
[ind1, ind2] = ind2sub([n1 n2], (1:n)');
ind_text = strcat(num2str(ind1), ',', num2str(ind2));
ind_sub = ind_text(S, :);

% random projection to 3-dim
% R = (rand(size(Msub,1),3) > 0.5)*2 - 1;  % random binary vectors
R = randn(size(Msub,1),3);  % random Gaussian vectors
V = Msub*R;
figure, scatter3(V(:,1), V(:,2), V(:,3)), xlim([-1 1]), ylim([-1 1]), hold on
text(V(:,1), V(:,2)+0.05, V(:,3), ind_sub), title('Random projection')
