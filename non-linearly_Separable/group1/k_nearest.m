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
actual_class(1:size(total_test,1),1:1) = 0;
actual_class(1:60,1) = 1;
actual_class(61:180,1) = 2;
actual_class(181:340,1) = 3;
%% Code
predicted_class(1:size(total_test,1),1:1) = 0;
accuracy(1:size(total_train,1),1:1) = 0;
k_value(1:size(total_train,1),1:1) = 0;
for k =ceil(sqrt(size(total_train,1)))%1:size(total_train,1)
%     k = number of nearest neighbours
    for p = 1:size(total_test,1)
        x = total_test(p,:);
        distance(1:size(total_train,1),1:1) = 0;
        for q=1:size(total_train,1)
            x_diff = x(1,1)-total_train(q,1);
            y_diff = x(1,2)-total_train(q,2);
            distance(q,1) = sqrt((x_diff*x_diff)+(y_diff*y_diff));
        end
        [B,I] = sort(distance);
    %     ascending sort i.e., from small distance to large distance
        nn_ids = I(1:k,1);

        count(1:3,1:1) = 0;
        for r = 1:k
            if (1 <= nn_ids(r,1)) == 1 &&  (nn_ids(r,1) <= 150) == 1
                count(1,1) = count(1,1)+1;
            elseif (151 <= nn_ids(r,1)) == 1 &&  (nn_ids(r,1) <= 450) == 1
                count(2,1) = count(2,1)+1;
            elseif (451 <= nn_ids(r,1)) == 1 &&  (nn_ids(r,1) <= 850) == 1
                count(3,1) = count(3,1)+1;
            end
        end

        [B2,I2] = sort(count);
        % ascending order sorting
        predicted_class(p,1) = I2(end,1);
    end
    [C,order] = confusionmat(actual_class,predicted_class);
    accuracy(k,1) = (sum(diag(C))./size(total_test,1))*100;
    k_value(k,1) = k;
end
%% Plot
% plot(k_value,accuracy,'Color','b','LineWidth',1.5)
% axis([0 (size(total_train,1)+100) 0 110])
% xlabel('K value')
% ylabel('% Accuracy')
% grid on
scatter(train1(:,1),train1(:,2),'red')
hold on
scatter(train2(:,1),train2(:,2),'blue','+')
hold on
scatter(train3(:,1),train3(:,2),'green','*')
hold on
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
        distance(1:size(total_train,1),1:1) = 0;
        for q=1:size(total_train,1)
            x_diff = point(1,1)-total_train(q,1);
            y_diff = point(1,2)-total_train(q,2);
            distance(q,1) = sqrt((x_diff*x_diff)+(y_diff*y_diff));
        end
        [B,I] = sort(distance);
    %     ascending sort i.e., from small distance to large distance
        nn_ids = I(1:k,1);

        count(1:4,1:1) = 0;
        for r = 1:k
            if (1 <= nn_ids(r,1)) == 1 &&  (nn_ids(r,1) <= 150) == 1
                count(1,1) = count(1,1)+1;
            elseif (151 <= nn_ids(r,1)) == 1 &&  (nn_ids(r,1) <= 450) == 1
                count(2,1) = count(2,1)+1;
            elseif (451 <= nn_ids(r,1)) == 1 &&  (nn_ids(r,1) <= 850) == 1
                count(3,1) = count(3,1)+1;
            end
        end

        [B2,I2] = sort(count);
        % ascending order sorting
        I = I2(end,1);

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