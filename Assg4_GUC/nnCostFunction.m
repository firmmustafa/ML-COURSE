function [J, grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m


% Encoding Layer 1
X = [ ones(length(X),1) X ];
a1= X;
a_prev = a1;
% Encoding Layer 2
z2 = Theta1 * a_prev';
a_feed = sigmoid(z2);
a2 = [ ones(length(a_feed'),1) a_feed' ];
% Encoding Layer 3
a_prev = a2;
z3 = Theta2 * a_prev';
a3 = sigmoid(z3);
% Cost Fn 
output = zeros(m,num_labels);
 for i = 1:m
     c=1;
     while y(i) ~= c
         c=c+1;
     end
     output( i , c) = 1;
 end 
 
 J =(1/m) * sum(sum((-output.*log((a3'))-((1-output).*log(1-(a3')))))); % without reg
 reg = (lambda/(2*m)) * (sum(sum(Theta1(:,2:end).^2)) + sum(sum(Theta2(:,2:end).^2)));
 J = J +  reg ; % with reg
 

%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.

for i = 1:m
    a1 = X(i,:); % 1x401
    z2 = Theta1*a1'; % 25x401 * 401x1 = 25*1
    a2_f = sigmoid(z2); % 25x1
    a2 = [1 a2_f']; %1x26
    z3 = Theta2 * a2'; % 10x26 * 26x1 = 10x1
    a3= sigmoid(z3);
    temp = zeros(length(a3),1);
    c=1;
    while y(i)~= c
     c=c+1;
    end 
    temp(c)=1;
    delta3 = a3 - temp; %10x1
    sig_grad = [ 1 sigmoidGradient(z2)']; % 1x26 
    delta2 = (Theta2'* delta3) .* sig_grad' ; % 26x1 * 1x26
    delta2 = delta2(2:end); % 25x1
    
    
    Theta1_grad = Theta1_grad + (delta2*a1); % 25x401
    Theta2_grad = Theta2_grad + (delta3*a2);  % 10x26
end

Theta1_grad = (1/m)*Theta1_grad ; % 25x401 
Theta2_grad = (1/m)*Theta2_grad ;  % 10x26

Theta1_ = (lambda/m)*[zeros(size(Theta1, 1),1) Theta1(:,2:end)]; % 25 x 401
Theta2_ = (lambda/m)*[zeros(size(Theta2, 1),1) Theta2(:,2:end)]; % 10 x 26
  
  %Adding regularization term to earlier calculated Theta_grad
  Theta1_grad = Theta1_grad + Theta1_;
  Theta2_grad = Theta2_grad + Theta2_;
  
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
