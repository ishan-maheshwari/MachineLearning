function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta

z = X*theta;

h = sigmoid(z);

h_minus = 1 - h;
y_minus = 1 - y;

J = (-1)/m;
J = J*((y')*log(h) + (y_minus')*log(h_minus));

theta_square = theta .^2;
theta_square = theta_square *lambda/(2*m);

for i=2:size(theta)
J = J + theta_square(i,1);
end

error = h - y;
grad = (X')*error;
grad = grad/m;

for i=2:size(theta)
grad(i,1) = grad(i,1) + (lambda/m)*theta(i,1);
end




% =============================================================

end
