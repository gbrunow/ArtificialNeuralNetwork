clear all;
clc;
delete(findall(0,'Type','figure'));
bdclose('all');

load('training.mat');
load('validation.mat');

input = tr_features;
desired_out = tr_labels;

%neural network settings
learning_rate = 0.0001;
alpha = 0.7;
hidden_neurons = 26;

input = [input ones(size(input,1),1) ]; 

%input layer weights
w1 = 0.5*(1-2*rand(size(input,2),hidden_neurons-1)); 

%hidden layer weights
w2 = 0.5*(1-2*rand(hidden_neurons,size(desired_out,2)));

%variables initialization        
epoch = 0;
error_trace = [];
error = inf;
dw1_previous = zeros(size(w1)); 
dw2_previous = zeros(size(w2)); 
figure;
tic
while error > 750
    hidden = [1./(1+exp(-input * w1)) ones(size(input,1),1)];
    output = 1./(1+exp(-hidden * w2));
    output_error = desired_out - output;
    
    error = trace(output_error'*output_error);
    error_trace = [error_trace error];
    
    deltas_out = output_error .* output .* (1-output);
    deltas_hid = deltas_out*w2' .* hidden .* (1-hidden);
    deltas_hid(:,size(deltas_hid,2)) = []; 
    
    dw1 = learning_rate * input' * deltas_hid + alpha * dw1_previous;   
    dw2 = learning_rate * hidden' * deltas_out + alpha * dw2_previous;
    w1 = w1 + dw1; 
    w2 = w2 + dw2;
    dw1_previous = dw1;
    dw2_previous = dw2;
    
    epoch = epoch + 1;
    if rem(epoch,50)==0
        
        right = 0;
        for i = 1:size(val_features,1)
            weighted_input = [val_features(i,:) 1]*w1;
            weighted_hidden = [1./(1+exp(-weighted_input)) 1]*w2;
            output = 1./(1+exp( - weighted_hidden));
%             output(output > 0.5) = 1;
%             output(output < 0.5) = 0;
            [~, index] = max(output);
            output = zeros(size(output));
            output(index) = 1;
            if output == val_labels(i,:)
               right = right + 1;
            end
        end
        percentage(epoch/50) = right/size(val_labels,1);
        subplot(2,1,1);
        plot(error_trace);
        xlabel('Epochs');
        ylabel('Error');
        subplot(2,1,2);
        plot(percentage, 'r');
        xlabel('Epochs');
        ylabel('Hits');
        drawnow;
        clc;
        toc
    end
end
toc

right = 0;
for i = 1:size(val_features,1)
    weighted_input = [val_features(i,:) 1]*w1;
    weighted_hidden = [1./(1+exp(-weighted_input)) 1]*w2;
    output = 1./(1+exp( - weighted_hidden));
%             output(output > 0.5) = 1;
%             output(output < 0.5) = 0;
    [~, index] = max(output);
    output = zeros(size(output));
    output(index) = 1;
    if output == val_labels(i,:)
       right = right + 1;
    end
end
percentage = 100*right/size(val_labels,1);
disp([' Correctness ' num2str(percentage) '%']);
