format long

result_10 = main_function(10)
result_20 = main_function(20)
result_40 = main_function(40)


function [A] = init_matrix(x)
% This function initializes the matrix as specified by the problem
A = zeros(x, x);
for i = 1:x
    for j = 1:x
        A(i, j) = (i^2 + j^2)^(0.5);
    end
end
end


function [off_diag_sum] = to_sum(A)
% This function compututes the square root of the sum of the squares of 
% the off-diagonal elements of the matrix
B = triu((A.^2), 1);
off_diag_sum = (2 * sum(B, 'all'))^(0.5);
end


function [A] = jacobi_conjugation(A, p, q)
% This function finds R and computes R'AR
phi = 0.5 * atan((2 * A(p, q)) ./ (A(q, q) - A(p, p)));
R = eye(length(A));
R(p, p) = cos(phi);
R(p, q) = sin(phi);
R(q, p) = -sin(phi);
R(q, q) = cos(phi);
A = R' * A * R;
end


function [new_A] = do_jacobi(A)
% This function finds the largest off-diagonal element in absolute 
% value and then apply the rotation
upper_A = abs(triu(A, 1));
[p, q] = find(upper_A==max(max(upper_A)));
[A] = jacobi_conjugation(A, p, q);
new_A = A;
end


function [result] = main_function(x)
% This is the main function that applies Jacobi's method to a matrix
% of given dimensionality.
max_iter = 10000;
matrices = cell(max_iter, 1);

% initialize matrix
[A_x] = init_matrix(x);
matrices{1, 1} = A_x;

% The for loop below does the iteration in Jacobi's method
for k = 1:max_iter-1
    A = matrices{k};
    new_A = do_jacobi(A);
    
    k = k + 1;
    matrices{k} = new_A;
    
    [off_diag_sum] = to_sum(new_A);
    if (off_diag_sum < 10^(-8))
        break
    end
end

not_rounded = diag(matrices{k});
% round the result to 8 significant digits
result = vpa(not_rounded, 8);
end

    


