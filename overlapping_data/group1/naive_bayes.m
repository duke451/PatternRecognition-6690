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
train4 = importdata('class4_train.txt');
test4 = importdata('class4_test.txt');

total_train = [train1; train2; train3; train4];
total_test = [test1; test2; test3; test4];
actual_class(1:size(total_test,1),1:1) = 0;
actual_class(1:100,1) = 1;
actual_class(101:200,1) = 2;
actual_class(201:300,1) = 3;
actual_class(301:400,1) = 4;
%% Code
mean1 = mean(train1);
% mean1 will be row vector containing the mean of each column
mean2 = mean(train2);
mean3 = mean(train3);
mean4 = mean(train4);

cov1 = cov(train1);
% C = cov(A)
% If A is a matrix whose columns represent random variables and whose rows 
% represent observations, C is the covariance matrix with the corresponding
% column variances along the diagonal.
cov2 = cov(train2);
cov3 = cov(train3);
cov4 = cov(train4);

predicted_class(1:size(total_test,1),1:1) = 0;
for p = 1:size(total_test,1)
    x = total_test(p,:);
    pxy1 = 1;
    for j = 1:size(x,2)
        first_term = 1./sqrt(2*pi*cov1(j,j));
        second_term = -0.5*(x(1,j)-mean1(1,j))^2 ./(cov1(j,j));
        pxy1 = pxy1*(first_term)*exp(second_term);
    end
    pxy1 = pxy1*(size(train1,1)./size(total_train,1));

    pxy2 = 1;
    for j = 1:size(x,2)
        first_term = 1./sqrt(2*pi*cov2(j,j));
        second_term = -0.5*(x(1,j)-mean2(1,j))^2 ./(cov2(j,j));
        pxy2 = pxy2*(first_term)*exp(second_term);
    end
    pxy2 = pxy2*(size(train2,1)./size(total_train,1));

    pxy3 = 1;
    for j = 1:size(x,2)
        first_term = 1./sqrt(2*pi*cov3(j,j));
        second_term = -0.5*(x(1,j)-mean3(1,j))^2 ./(cov3(j,j));
        pxy3 = pxy3*(first_term)*exp(second_term);
    end
    pxy3 = pxy3*(size(train3,1)./size(total_train,1));

    pxy4 = 1;
    for j = 1:size(x,2)
        first_term = 1./sqrt(2*pi*cov4(j,j));
        second_term = -0.5*(x(1,j)-mean4(1,j))^2 ./(cov4(j,j));
        pxy4 = pxy4*(first_term)*exp(second_term);
    end
    pxy4 = pxy4*(size(train4,1)./size(total_train,1));

    pxy = [pxy1;pxy2;pxy3;pxy4];
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
hold on
scatter(train4(:,1),train4(:,2),'magenta','s')
hold on
legend('class1', 'class2','class3', 'class4','Location','Best');

%% Decision region
xrange(1:300,1) = 0;
yrange(1:300,1) = 0;
color_matrix(1:300,1:3) = 0;
i = 1;
for xvalue = -6:0.1:10
    for yvalue = -15:0.1:15
        xrange(i,1) = xvalue;
        yrange(i,1) = yvalue;
        point = [xvalue yvalue];
        pxy1 = 1;
        for j = 1:size(point,2)
            first_term = 1./sqrt(2*pi*cov1(j,j));
            second_term = -0.5*(point(1,j)-mean1(1,j))^2 ./(cov1(j,j));
            pxy1 = pxy1*(first_term)*exp(second_term);
        end
        pxy1 = pxy1*(size(train1,1)./size(total_train,1));

        pxy2 = 1;
        for j = 1:size(point,2)
            first_term = 1./sqrt(2*pi*cov2(j,j));
            second_term = -0.5*(point(1,j)-mean2(1,j))^2 ./(cov2(j,j));
            pxy2 = pxy2*(first_term)*exp(second_term);
        end
        pxy2 = pxy2*(size(train2,1)./size(total_train,1));

        pxy3 = 1;
        for j = 1:size(point,2)
            first_term = 1./sqrt(2*pi*cov3(j,j));
            second_term = -0.5*(point(1,j)-mean3(1,j))^2 ./(cov3(j,j));
            pxy3 = pxy3*(first_term)*exp(second_term);
        end
        pxy3 = pxy3*(size(train3,1)./size(total_train,1));

        pxy4 = 1;
        for j = 1:size(point,2)
            first_term = 1./sqrt(2*pi*cov4(j,j));
            second_term = -0.5*(point(1,j)-mean4(1,j))^2 ./(cov4(j,j));
            pxy4 = pxy4*(first_term)*exp(second_term);
        end
        pxy4 = pxy4*(size(train4,1)./size(total_train,1));

        pxy = [pxy1;pxy2;pxy3;pxy4];
        [~,I] = max(pxy);
        if I == 1
           color_matrix(i,:) = [1 0 0];%red
        else
            if I == 2
                color_matrix(i,:) = [0 0 1];%blue
            else
                if I == 3
                    color_matrix(i,:) = [0 1 0];%green
                else 
                    color_matrix(i,:) = [1 0 1];%magenta
                end
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