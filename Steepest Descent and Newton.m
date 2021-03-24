% Problem 3
rng('default');
rng(2021);
set(0, 'DefaultLineLineWidth', 2);
format long

% We randomly generate A, b, c, and x_0 such that x_0 is always in the 
% domain of function f.
x_0 = randn(100, 1);
A = randn(500, 100);
b = A*x_0 + 200*rand(500, 1);
c = randn(100, 1);

% Part b
%{
As an example, we set alpha_0 = 1, c = 0.01, rho = 0.05, and epsilon
=10^(-5). We plot the log of error against k and plot the error against k.
Below is the code for Newton's Method.
%}

% run the steepest descent algorithm
[e_1 x_1 count_1] = steepest_descent(A, b, c, x_0, 0.01, 0.05, 10^(-5), 1);
error_1 = e_1(1:count_1);

% plot of error vs. k
plot(1:count_1, error_1);
title('Part b Steepest Descent: error vs. k', 'fontsize', 15)
xlabel('k', 'fontsize', 15)
ylabel('$e_{k}$', 'fontsize', 15, 'interpreter', 'latex')
fig = gcf;
exportgraphics(fig, 'partb_Steepest_Descent_error.png')

% plot of log error vs. k
plot(1:count_1, log(error_1));
title('Part b Steepest Descent: log error vs. k', 'fontsize', 15)
xlabel('k', 'fontsize', 15)
ylabel('$\log(e_{k})$', 'fontsize', 15, 'interpreter', 'latex')
fig = gcf;
exportgraphics(fig, 'partb_Steepest_Descent_logerror.png')

%{
As an example, we set alpha_0 = 1, c = 0.01, rho = 0.05, and epsilon
=10^(-5). We plot the log of error against k and plot the error against k.
Below is the code for Newton's Method
%}

% run the newton's method
[e_1 x_1 count_1] = Newton(A, b, c, x_0, 0.01, 0.05, 10^(-5), 1);
error_1 = e_1(1:count_1);

% plot of error vs. k
plot(1:count_1, error_1);
title('Part b Newton: error vs. k', 'fontsize', 15)
xlabel('k', 'fontsize', 15)
ylabel('$e_{k}$', 'fontsize', 15, 'interpreter', 'latex')
fig = gcf;
exportgraphics(fig, 'partb_Newton_error.png')

% plot of log error vs. k
plot(1:count_1, log(error_1));
title('Part b Newton: log error vs. k', 'fontsize', 15)
xlabel('k', 'fontsize', 15)
ylabel('$\log(e_{k})$', 'fontsize', 15, 'interpreter', 'latex')
fig = gcf;
exportgraphics(fig, 'partb_Newton_logerror.png')

% part c
%{
First, we vary c_search while holding other parameters constant. We 
set alpha_0 = 1, rho = 0.05, and epsilon =10^(-3). Below is the code
for the Steepest Descent Method.
%}

% run the steepest descent algorithm
index = 1;
iteration = zeros(7, 1);
error = cell(7, 1);
for c_search = [0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90]
    [e x count] = steepest_descent(A, b, c, x_0, c_search, 0.05,...
        10^(-3), 1);
    error{index} = log(e(1:count));
    iteration(index) = count;
    index = index + 1;
end

% error plot for different c
plot(1:iteration(1), error{1}, 'r', 1:iteration(2), error{2}, 'g',...
    1:iteration(3), error{3}, 'b', 1:iteration(4), error{4}, 'c',...
    1:iteration(5), error{5}, 'm', 1:iteration(6), error{6}, 'y',...
    1:iteration(7), error{7}, 'k');
legend('c=0.01', 'c=0.05', 'c=0.10', 'c=0.25', 'c=0.50', 'c=0.75',...
    'c=0.90')
title('Part c Steepest Descent: log error vs. k', 'fontsize', 15)
xlabel('k', 'fontsize', 15)
ylabel('$\log(e_{k})$', 'fontsize', 15, 'interpreter', 'latex')
fig = gcf;
exportgraphics(fig, 'partc_Steepest_Descent_c_search.png')




%{
First, we vary c_search while holding other parameters constant. We 
set alpha_0 = 1, rho = 0.05, and epsilon =10^(-5). Below is the code
for Newton's Method.
%}

% run the newton's method
index = 1;
iteration = zeros(7, 1);
error = cell(7, 1);
for c_search = [0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90]
    [e x count] = Newton(A, b, c, x_0, c_search, 0.05, 10^(-5), 1);
    error{index} = log(e(1:count));
    iteration(index) = count;
    index = index + 1;
end

% error plot for different c
plot(1:iteration(1), error{1}, 'r', 1:iteration(2), error{2}, 'g',...
    1:iteration(3), error{3}, 'b', 1:iteration(4), error{4}, 'c',...
    1:iteration(5), error{5}, 'm', 1:iteration(6), error{6}, 'y',...
    1:iteration(7), error{7}, 'k');
legend('c=0.01', 'c=0.05', 'c=0.10', 'c=0.25', 'c=0.50', 'c=0.75',...
    'c=0.90')
title('Part c Newton: log error vs. k', 'fontsize', 15)
xlabel('k', 'fontsize', 15)
ylabel('$\log(e_{k})$', 'fontsize', 15, 'interpreter', 'latex')
fig = gcf;
exportgraphics(fig, 'partc_Newton_c_search.png')

%{
Next, we vary rho while holding other parameters constant. We 
set alpha_0 = 1, c_search = 0.9, and epsilon =10^(-3). Below is the code
for the Steepest Descent Method.
%}

%run the steepest descent algorithm
index = 1;
iteration = zeros(5, 1);
error = cell(5, 1);
for rho = [0.05, 0.25, 0.50, 0.75, 0.95]
    [e x count] = steepest_descent(A, b, c, x_0, 0.9, rho, 10^(-3), 1);
    error{index} = log(e(1:count));
    iteration(index) = count;
    index = index + 1;
end

% error plot for different rho
plot(1:iteration(1), error{1}, 'r', 1:iteration(2), error{2}, 'g',...
    1:iteration(3), error{3}, 'b', 1:iteration(4), error{4}, 'c',...
    1:iteration(5), error{5}, 'm');
legend('rho=0.05', 'rho=0.25', 'rho=0.50', 'rho=0.75', 'rho=0.95')
title('Part c Steepest Descent: log error vs. k', 'fontsize', 15)
xlabel('k', 'fontsize', 15)
ylabel('$\log(e_{k})$', 'fontsize', 15, 'interpreter', 'latex')
fig = gcf;
exportgraphics(fig, 'partc_Steepest_Descent_rho.png')

%{
Next, we vary rho while holding other parameters constant. We 
set alpha_0 = 1, c_search = 0.05, and epsilon =10^(-5). Below is the code
for Newton's Method.
%}

% run the newton's method
index = 1;
iteration = zeros(5, 1);
error = cell(5, 1);
for rho = [0.05, 0.25, 0.50, 0.75, 0.95]
    [e x count] = Newton(A, b, c, x_0, 0.05, rho, 10^(-5), 1);
    error{index} = log(e(1:count));
    iteration(index) = count;
    index = index + 1;
end

% error plot for different rho
plot(1:iteration(1), error{1}, 'r', 1:iteration(2), error{2}, 'g',...
    1:iteration(3), error{3}, 'b', 1:iteration(4), error{4}, 'c',...
    1:iteration(5), error{5}, 'm');
legend('rho=0.05', 'rho=0.25', 'rho=0.50', 'rho=0.75', 'rho=0.95')
title('Part c Newton: log error vs. k', 'fontsize', 15)
xlabel('k', 'fontsize', 15)
ylabel('$\log(e_{k})$', 'fontsize', 15, 'interpreter', 'latex')
fig = gcf;
exportgraphics(fig, 'partc_Newton_rho.png')

%{
Lastly, we vary epsilon while holding other parameters constant. We 
set alpha_0 = 1, c_search = 0.05, rho = 0.05. Below is the code
for the Steepest Descent Method.
%}

% run the steepest descent algorithm
index = 1;
iteration = zeros(3, 1);
error = cell(3, 1);
for epsilon = [10^(-3), 10^(-5), 10^(-8)]
    [e x count] = steepest_descent(A, b, c, x_0, 0.05, 0.05, epsilon, 1);
    error{index} = log(e(1:count));
    iteration(index) = count;
    index = index + 1;
end

% error plot for different epsilon
plot(1:iteration(1), error{1}, 'r', 1:iteration(2), error{2}, 'g',...
    1:iteration(3), error{3}, 'b');
legend('epsilon=0.01', 'epsilon=0.00001', 'epsilon=0.00000001')
title('Part c Steepest Descent: log error vs. k', 'fontsize', 15)
xlabel('k', 'fontsize', 15)
ylabel('$\log(e_{k})$', 'fontsize', 15, 'interpreter', 'latex')
fig = gcf;
exportgraphics(fig, 'partc_Steepest_Descent_epsilon.png')

%{
Lastly, we vary epsilon while holding other parameters constant. We 
set alpha_0 = 1, c_search = 0.90, rho = 0.5. Below is the code
for Newton's Method.
%}

% run the newton's method
index = 1;
iteration = zeros(3, 1);
error = cell(3, 1);
for epsilon = [10^(-3), 10^(-5), 10^(-8)]
    [e x count] = Newton(A, b, c, x_0, 0.90, 0.5, epsilon, 1);
    error{index} = log(e(1:count));
    iteration(index) = count;
    index = index + 1;
end

% error plot for different epsilon
plot(1:iteration(1), error{1}, 'r', 1:iteration(2), error{2}, 'g',...
    1:iteration(3), error{3}, 'b');
legend('epsilon=0.01', 'epsilon=0.00001', 'epsilon=0.00000001')
title('Part c Newton: log error vs. k', 'fontsize', 15)
xlabel('k', 'fontsize', 15)
ylabel('$\log(e_{k})$', 'fontsize', 15, 'interpreter', 'latex')
fig = gcf;
exportgraphics(fig, 'partc_Newton_epsilon.png')

%{
Now, we want to repeat part b without doing line search. As an example, 
we set alpha_0 = 1, c = 0.90, rho = 0.95, and epsilon = 10^(-5). We note, 
however, that doing steepest descent without line search will not give
us a final result. We thus comment out the line of code below. The Newton's
Method will work, and we plot log error against k for Newton's Method.
%}

%[e x count] = steepest_descent2(A, b, c, x_0, 0.01, 0.05, 10^(-5), 1);

% run the newton's method, once with line search, and once without line
% search
[e x count] = Newton2(A, b, c, x_0, 0.90, 0.95, 10^(-5), 1);
[e_BTLS x_BTLS count_BTLS] = Newton(A, b, c, x_0, 0.90, 0.95, 10^(-5), 1);
error = e(1:count);
error_BTLS = e_BTLS(1:count_BTLS);

% error plot that compares newton's method with line search and without 
% line search
plot(1:count, log(error), 'r', 1:count_BTLS, log(error_BTLS), 'b');
title('Part c Newton: log error vs. k', 'fontsize', 15)
xlabel('k', 'fontsize', 15)
ylabel('$\log(e_{k})$', 'fontsize', 15, 'interpreter', 'latex')
legend('Without BTLS', 'With BTLS')
fig = gcf;
exportgraphics(fig, 'partc_Newton_BTLS.png')

% Below are the main functions implementing the steepest descent algorithm 
% and newton's method

function result = f(A, b, c, x)
%{
This function defines our objective function.
A, b, c are the parameters of the function.
x is the point at which the function is evaluated.
%}
    result = transpose(c)*x - sum(log(b-A*x));
end

function result = grad(A, b, c, x)
%{
This function computes the gradient of our objective function 
A, b, c are the parameters of the function.
x is the point at which the function is evaluated.
%}

    result = zeros(100, 1);
    bottom = b - A*x;
    % we loop through every position of the result vector and calculate 
    % value separately.
    for i = 1:100
        sum = c(i);
        for k = 1:500
            sum = sum + A(k, i)/bottom(k);
        end
        % we assign the value to result
        result(i) = sum;
    end
end


function result = hess(A, b, c, x)
%{
This function computes the Hessian of our objective function.
A, b, c are the parameters of the function.
x is the point at which the function is evaluated.
%}
    result = zeros(100, 100);
    bottom = (b - A*x).^2;
    % we loop through every position of the result matrix and calculate 
    % value separately.
    for i = 1:100
        for j = 1:100
            v1 = A(:, i);
            v2 = A(:, j);
            top = v1 .* v2;
        
            sum = 0;
            % below is the formula for the value at each (i, j) position
            % of the hessian.
            for k = 1:500
                sum = sum + top(k)/bottom(k);
            end
            % we assign the value to result
            result(i, j) = sum;
        end
    end
            
end


function alpha_k = line_search(A, b, c, x_k, p_k, c_search, rho, alpha_0)
%{
This function implements the backtracking line search.
A, b, c are the parameters of the function.
x_k is the point at which the function is evaluated.
p_k is the search direction.
c_search and rho are backtracking line search parameters.
alpha_0 is the initial step size
%}
    alpha = alpha_0;
    g = grad(A, b, c, x_k);
    x_new = x_k + alpha*p_k;
    
    % The following loop makes sure that the updated x_new is always
    % in the domain of the function.
    while any(b - A*x_new <= 0)
        alpha=alpha * rho;
        x_new = x_k + alpha * p_k;
    end
    
    % The following loop makes sure that the Armijo condition is met. 
    armijo = f(A, b, c, x_k) + c_search*alpha*transpose(g)*p_k;
    while f(A, b, c, x_new) > armijo
        alpha = rho*alpha;
        x_new = x_k + alpha*p_k;
        armijo = f(A, b, c, x_k) + c_search*alpha*transpose(g)*p_k;
    end
    
    % We return the final step size.
    alpha_k = alpha;
    
end


function [e x count] = steepest_descent(A, b, c, x_0, c_search, rho,...
    epsilon, alpha_0)
%{
This function implements the steepest descent algorithm.
A, b, c are the parameters of the function.
x_0 is the starting point.
c_search and rho are backtracking line search parameters.
epsilon is the tolerance for the stopping conditions.
alpha_0 is the initial step size
%}
    % we first initialize data structures to store our results.
    % e is the error.
    max_iter = 100000;
    count = 1;
    p = cell(max_iter, 1);
    x = cell(max_iter, 1);
    alpha = zeros(max_iter, 1);
    e = zeros(max_iter, 1);
    e(1) = f(A, b, c, x_0);
    x{1} = x_0;
    
    % The following loop is the main body of the steepest descent algorithm
    for k = 1:max_iter
        % first find the serch direction
        p{k} = -grad(A, b, c, x{k});
        % next find the step size through BTLS
        alpha(k) = line_search(A, b, c, x{k}, p{k}, c_search,...
            rho, alpha_0);
        % update x
        x{k+1} = x{k} + alpha(k) * p{k};
        % store the error
        e(k+1) = f(A, b, c, x{k+1});
        
        count = count + 1;
        
        % This is the stopping condition
        if norm(grad(A, b, c, x{k+1})) < epsilon
            break;
        end
        
        % It is not feasible to reach the accuracy of 10^(-8).
        % For epsilon = 10^(-8), we terminate the program whenever
        % the norm of the gradient stops decreasing. 
        if epsilon == 10^(-8) & norm(grad(A, b, c, x{k+1})) ==...
                norm(grad(A, b, c, x{k}))
            break;
        end
                
    end
    
    % we compute the error.
    e = e - f(A, b, c, x{count});
end

function [e x count] = Newton(A, b, c, x_0, c_search, rho,...
    epsilon, alpha_0)
%{
This function implements the newton's method.
A, b, c are the parameters of the function.
x_0 is the starting point.
c_search and rho are backtracking line search parameters.
epsilon is the tolerance for the stopping conditions.
alpha_0 is the initial step size
%}
    
    % we first initialize data structures to store our results.
    % e is the error.
    max_iter = 100000;
    count = 1;
    p = cell(max_iter, 1);
    x = cell(max_iter, 1);
    alpha = zeros(max_iter, 1);
    e = zeros(max_iter, 1);
    e(1) = f(A, b, c, x_0);
    x{1} = x_0;
    
    % The following loop is the main body of the newton's method
    for k = 1:max_iter
        g = grad(A, b, c, x{k});
        H = hess(A, b, c, x{k});
        % we find the search direction
        p{k} = -inv(H)*g;
        % we compute lambda_sq
        lambda_sq = transpose(g)*inv(H)*g;
        
        % This is the stopping condition
        if lambda_sq/2 <= epsilon
            break;
        end
        
        count = count + 1;
        
        % find the step size
        alpha(k) = line_search(A, b, c, x{k}, p{k}, c_search,...
            rho, alpha_0);
        % update x
        x{k+1} = x{k} + alpha(k)*p{k};
        % store the error
        e(k+1) = f(A, b, c, x{k+1});
    end
    
    % we compute the error.
    e = e - f(A, b, c, x{count});
end


function [e x count] = steepest_descent2(A, b, c, x_0, c_search,...
    rho, epsilon, alpha_0)
%{
This function implements the steepest descent algorithm without BTLS
A, b, c are the parameters of the function.
x_0 is the starting point.
c_search and rho are backtracking line search parameters.
epsilon is the tolerance for the stopping conditions.
alpha_0 is the initial step size. I will not comment heavily since
we jsut take out the BTLS step.
%}
    
    max_iter = 100000;
    count = 1;
    p = cell(max_iter, 1);
    x = cell(max_iter, 1);
    % note that alpha is a fixed value.
    alpha = alpha_0;
    e = zeros(max_iter, 1);
    e(1) = f(A, b, c, x_0);
    x{1} = x_0;
    
    for k = 1:max_iter
        
        p{k} = -grad(A, b, c, x{k});
        x{k+1} = x{k} + 1 * p{k};
        e(k+1) = f(A, b, c, x{k+1})
        
        count = count + 1;
        
        if norm(grad(A, b, c, x{k+1})) < epsilon
            break;
        end
    end
    
    e = e - f(A, b, c, x{count});
end


function [e x count] = Newton2(A, b, c, x_0, c_search, rho,...
    epsilon, alpha_0)
%{
This function implements the newton's method without BTLS
A, b, c are the parameters of the function.
x_0 is the starting point.
c_search and rho are backtracking line search parameters.
epsilon is the tolerance for the stopping conditions.
alpha_0 is the initial step size
%}
    max_iter = 100000;
    count = 1;
    p = cell(max_iter, 1);
    x = cell(max_iter, 1);
    alpha = ones(max_iter, 1);
    e = zeros(max_iter, 1);
    e(1) = f(A, b, c, x_0);
    x{1} = x_0;
    for k = 1:max_iter
        g = grad(A, b, c, x{k});
        H = hess(A, b, c, x{k});

        p{k} = -inv(H)*g;
        lambda_sq = transpose(g)*inv(H)*g;
        
        if lambda_sq/2 <= epsilon
            break;
        end
        
        % To make sure that x is close enough to the actual minimizer,
        % we do BTLS for the first 20 steps, after which we switch to 
        % newton's method without BTLS.
        if k < 20
            alpha(k) = line_search(A, b, c, x{k}, p{k}, c_search,...
                rho, alpha_0);
        end
        
        x{k+1} = x{k} + alpha(k)*p{k};
        e(k+1) = f(A, b, c, x{k+1});
        
        count = count + 1;
    end
    
    e = e - f(A, b, c, x{count});
end

