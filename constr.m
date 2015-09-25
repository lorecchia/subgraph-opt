clear all;
close all;
% cvx_solver sedumi

%% data generation, generate n1xn2 lattice, 4-connected
n1 = 5;
n2 = 5;
n = n1*n2;
K = 4;
xx_line = zeros(n,1);
% xx_line([7, 8, 9, 12, 13, 14, 19, 20, 24]) = 1;
xx_line([7, 9, 10, 15]) = 1;  % 4 nodes with value 1
% xx_line(9) = 1.1;
xx = reshape(xx_line, n1, n2);

% observations with no noise
yy = xx;
yy_line = xx_line; %0.01*randn(size(xx_line));

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
% gplot(Adj, xxg, '-o');
% axis([0, n2+1, 0, n1+1]);
% unnormalized graph Laplacian
d = sum(Adj,2);
D = diag(d);
L = D-Adj;

% plot using gplot, not used for now
% [xg, yg] = meshgrid(1:n1, 1:n2);
% xxg = [xg(:), yg(:)];

% figure, gplot(Adj, xxg, ':x');
% hold on;
% gplot(diag(xx_line)*Adj*diag(xx_line), xxg, 'r-o');
% axis([0, n2+1, 0, n1+1]);

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
        trace( M ) <= K
%         sum(sum(M)) <= K
        L_A_M - gamma*L_M == semidefinite(n)

        % constaints from NIPS paper
        M <= 1
        M(p,p) == 1
        for i = 1:n
            M(i,i) <= M(p,i)
            M(i,:) <= M(i,i)
        end
cvx_end

S = diag(M);% > 1e-3;

% show result using gplot, not used for now
% figure, gplot(Adj, xxg, ':x');
% hold on;
% gplot(diag(S)*Adj*diag(S), xxg, 'r-o');
% axis([0, n2+1, 0, n1+1]);

% show result using imagesc, no thresholding for now
figure, imagesc(reshape(S, n1, n2)), colorbar, title('Estimated');
