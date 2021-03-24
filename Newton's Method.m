% Problem 4
rng('default');
rng(1212);

% Part d(i)

% 2 * 2 matrix
fprintf("Result for a randomly generated 2 by 2 integer matrix:")
A = randi([-1000, 1000], 2);
A_inv = analytical_inv(A);
X_star = Newton_inv(A, 10^(-8));
forward_error = norm((X_star - A_inv), 'fro')

% 10 * 10 matrix
fprintf("Result for a randomly generated 10 by 10 diagonal matrix:")
diagonal = randn(10, 1);
while any(diagonal == 0)
    diagonal = randn(10, 1);
end
    
A = diag(diagonal);
A_inv = analytical_inv(A);
X_star = Newton_inv(A, 10^(-8));
forward_error = norm((X_star - A_inv), 'fro')

% Part d(ii)

for n = [10, 100, 1000]
    A = randn(n, n);
    Y_star = inv(A);
    X_star = Newton_inv(A, 10^(-8));
    fprintf("Result for a randomly generated %u by %u matrix:", n, n)
    error1 = norm((eye(n) - A*X_star), 'fro')
    error2 = norm((eye(n) - A*Y_star), 'fro')
end


function result = Newton_inv(A, epsilon)
%{
This function applies Newton's Method to compute the inverse of a matrix.
A is the input matrix whose inverse we would like to compute
epsilon is the tolerance for the stopping condition
%}

    % we initialize the x_0
    alpha_bound = 2/((norm(A, 'fro'))^2);
    alpha = rand*alpha_bound;
    X_0 = alpha*transpose(A);
    
    [n1 n2] = size(A);
    
    % we initialize the data structure where we store our results.
    max_iter = 100000;
    count = 1;
    X = cell(max_iter, 1);
    e = zeros(max_iter, 1);
    X{1} = X_0;
    
    % The for loop below implements the main algorithm
    for k = 1:max_iter
        X{k+1} = X{k}*(2*eye(n1) - A*X{k});
        e(k) = norm((X{k+1} - X{k}), 'fro');
        
        count = count + 1;
        % This is the stopping condition
        if e(k) <= epsilon
            break;
        end
    end
    
    % Lastly, we return the result.
    result = X{count};
end

function result = analytical_inv(A)
%{
This function analytically computes the inverse of a 2*2 matrix or a 
10*10 diagonal matrix.
A is the input matrix - it is either a 2*2 matrix or a 10*10 diagonal 
matrix.
%}
    [n1 n2] = size(A);
    % if the matrix is 2*2, we compute the inverse as follows
    if n1 == 2
        a = A(1,1);
        b = A(1, 2);
        c = A(2, 1);
        d = A(2, 2);
        
        result = [d, -b; -c, a];
        result = (1 / (a*d - b*c)) * result;
        
    end
    
    % if A is a 10*10 diagonal matrix, we compute the inverse as follows
    if n1 == 10
        result = zeros(10, 10);
        for i = 1:10
            result(i, i) = 1/A(i, i);
        end
        
    end
    

end


