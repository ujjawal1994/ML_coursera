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

z = sigmoid(X*(theta));
reg_term = 0;
for i =2:size(theta,1)
	reg_term+=theta(i)^2;
end
reg_term = reg_term*lambda/(2*m);
for i = 1:m
	temp = -1*((y(i)*log(z(i)))+((1-y(i))*log(1-z(i))));
	J = J+temp;
end
J = J/m + reg_term;
m;

for i = 1:m
	grad(1) += (z(i)-y(i))*X(i,1);
end
grad(1)/=m;

for j = 2:size(theta)(1);
	for i = 1:m
		grad(j) += (z(i)-y(i))*X(i,j);
	end
	grad(j)/=m;
	grad(j)+=(lambda*theta(j)/m) ;
end





% =============================================================

end
