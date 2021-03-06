clc;
clear;
close all;

%% Loading the data
train1 = importdata('class1_train.txt');
test1 = importdata('class1_test.txt');
train2 = importdata('class2_train.txt');
test2 = importdata('class2_test.txt');
train3 = importdata('class3_train.txt');
test3 = importdata('class3_test.txt');

total_train = [train1; train2; train3];
total_test = [test1; test2; test3];
d = size(total_train,2);
actual_class(1:size(total_test,1),1:1) = 0;
actual_class(1:60,1) = 1;
actual_class(61:180,1) = 2;
actual_class(181:340,1) = 3;
%% Code
mean1 = mean(train1);
mean2 = mean(train2);
mean3 = mean(train3);

cov1 = cov(train1);
cov2 = cov(train2);
cov3 = cov(train3);

predicted_class(1:size(total_test,1),1:1) = 0;
for p = 1:size(total_test,1)
    x = total_test(p,:);
    
    first_term = sqrt((2*pi).^d)*sqrt(det(cov1));
    second_term = -0.5*(x.' - mean1.').'/cov1*(x.' - mean1.');
    pxy1 = (1./first_term)*exp(second_term);
    pxy1 = pxy1*(size(train1,1)./size(total_train,1));
    
    first_term = sqrt((2*pi).^d)*sqrt(det(cov2));
    second_term = -0.5*(x.' - mean2.').'/cov2*(x.' - mean2.');
    pxy2 = (1./first_term)*exp(second_term);
    pxy2 = pxy2*(size(train2,1)./size(total_train,1));

    first_term = sqrt((2*pi).^d)*sqrt(det(cov3));
    second_term = -0.5*(x.' - mean3.').'/cov3*(x.' - mean3.');
    pxy3 = (1./first_term)*exp(second_term);
    pxy3 = pxy3*(size(train3,1)./size(total_train,1));

    pxy = [pxy1;pxy2;pxy3];
    [B2,I2] = sort(pxy);
    % ascending order sorting
    predicted_class(p,1) = I2(end,1);
end
%% Post processing
[C,order] = confusionmat(actual_class,predicted_class);
accuracy = (sum(diag(C))./size(total_test,1))*100;
scatter(train1(:,1),train1(:,2),'red')
hold on
scatter(train2(:,1),train2(:,2),'blue','+')
hold on
scatter(train3(:,1),train3(:,2),'green','*')
legend('class1', 'class2','class3','Location','Best');

%% Decision region plot
xrange(1:300,1) = 0;
yrange(1:300,1) = 0;
color_matrix(1:300,1:3) = 0;
i = 1;
for xvalue = -4:0.05:4
    for yvalue = -4:0.05:4
        xrange(i,1) = xvalue;
        yrange(i,1) = yvalue;
        point = [xvalue yvalue];
        first_term = sqrt((2*pi).^d)*sqrt(det(cov1));
        second_term = -0.5*(point.' - mean1.').'/cov1*(point.' - mean1.');
        pxy1 = (1./first_term)*exp(second_term);
        pxy1 = pxy1*(size(train1,1)./size(total_train,1));

        first_term = sqrt((2*pi).^d)*sqrt(det(cov2));
        second_term = -0.5*(point.' - mean2.').'/cov2*(point.' - mean2.');
        pxy2 = (1./first_term)*exp(second_term);
        pxy2 = pxy2*(size(train2,1)./size(total_train,1));

        first_term = sqrt((2*pi).^d)*sqrt(det(cov3));
        second_term = -0.5*(point.' - mean3.').'/cov3*(point.' - mean3.');
        pxy3 = (1./first_term)*exp(second_term);
        pxy3 = pxy3*(size(train3,1)./size(total_train,1));

        pxy = [pxy1;pxy2;pxy3];
        [~,I] = max(pxy);

        if I == 1
           color_matrix(i,:) = [1 0 0];%red
        else
            if I == 2
                color_matrix(i,:) = [0 0 1];%blue
            else
                color_matrix(i,:) = [0 1 0];%green
            end
        end
        i = i+1;
    end
end
% color_matrix_vector = char(color_matrix);
s = scatter(xrange,yrange,[],color_matrix,'filled');
s.Marker = 's';
% s.LineWidth = 0.01;
s.MarkerFaceAlpha = 0.05;
s.MarkerEdgeAlpha = 0.05;