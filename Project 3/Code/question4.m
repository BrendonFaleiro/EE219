%Importing the Data
filename = '..\Dataset\ml-100k\u.data';
delimiterIn = ('\t');
u= dlmread(filename, delimiterIn);

r=NaN(943,1682);
w=zeros(943,1682);
r1=r;
w1=w;
option.iter=50;
%Normal R and W
for i=1:100000
    r(u(i,1),u(i,2)) = u(i,3);
    w(u(i,1),u(i,2)) = 1;
end
%Swapped R and W
for i=1:100000
    r1(u(i,1),u(i,2)) = u(i,3);
    if isnan(r1(u(i,1),u(i,2))) == 0
        r1(u(i,1),u(i,2)) = 1;
    end
    w1(u(i,1),u(i,2))=u(i,3);
end
threshold = linspace(0.1, 5.0, 50); 
k=[10,50,100];
lambda=[0.01,0.1,1];
e=[0,0,0];
e1=[0,0,0];
e2=[0 0 0;0 0 0;0 0 0];
e3=[0 0 0;0 0 0;0 0 0];
precision_vector = zeros(50,1,3);
recall_vector = zeros(50,1,3);
precision_vector2 = zeros(50,1,3,3);
recall_vector2 = zeros(50,1,3,3);
precision_vector1 = zeros(50,1,3);
recall_vector1 = zeros(50,1,3);
precision_vector3 = zeros(50,1,3,3);
recall_vector3 = zeros(50,1,3,3);

for d=1:3
    for b=1:3
        [U,V,~,~,res]=wnmfrule(r,k(b),option); %Unregularized R and W
        [u1,v1,~,~,res1]=wnmfrule(r1,k(b),option); %Unregularized R and W Swapped
        [u2,v2,~,~,res2]=reg_wnmfrule(r,w,k(b),lambda(d),option);%Regularized R and W
        [u3,v3,~,~,res3]=reg_wnmfrule(r1,w1,k(b),lambda(d),option);%Regularized R and W Swapped
        p=U*V;
        p1=u1*v1;
        p2=u2*v2;
        p3=u3*v3;
        for i=1:943
            for j=1:1682
                if isnan(r(i,j))==0 | isnan(r1(i,j))==0
                    e(b)=e(b)+w(i,j)*(r(i,j)-p(i,j))^2;%Error for Unregularized R and W
                    e2(b,d)=e2(b,d)+w(i,j)*(r(i,j)-p2(i,j))^2;%Error for R and W Swapped
                    e1(b)=e1(b)+w1(i,j)*(r1(i,j)-p1(i,j))^2;%Error for Regularized R and W
                    e3(b,d)=e3(b,d)+w1(i,j)*(r1(i,j)-p3(i,j))^2;%Error for Regularized R and W Swapped
                end
            end
        end
        %Calculting the Precision and Recall for each of the following
        for th=1:50
            precision_vector(th,1,b)=length(find((p(:, :)>threshold(th)) & (r>3)))/length(find(p(:, :)>threshold(th)));
            recall_vector(th,1,b)=length(find((p(:, :)>threshold(th)) & (r>3)))/length(find(r>3));
            
            precision_vector2(th,1,b,d)=length(find((p2(:, :)>threshold(th)) & (r>3)))/length(find(p2(:, :)>threshold(th)));
            recall_vector2(th,1,b,d)=length(find((p2(:, :)>threshold(th)) & (r>3)))/length(find(r>3));
            
            precision_vector1(th,1,b)=length(find((p1(:, :)>threshold(th)) & (r1==1)))/length(find(p1(:, :)>threshold(th)));
            recall_vector1(th,1,b)=length(find((p1(:, :)>threshold(th)) & (r1==1)))/length(find(r1==1));
            
            precision_vector3(th,1,b,d)=length(find((p3(:, :)>threshold(th)) & (r1==1)))/length(find(p3(:, :)>threshold(th)));
            recall_vector3(th,1,b,d)=length(find((p3(:, :)>threshold(th)) & (r1==1)))/length(find(r1==1));
        end
    end
end

figure;
subplot(1,2,1);
plot(recall_vector(:,1,1), precision_vector(:,1,1),'r',recall_vector(:,1,2), precision_vector(:,1,2),'b',recall_vector(:,1,3), precision_vector(:,1,3),'g')
 title('ROC for Unregularized R and W')
 legend('k = 10', 'k = 50', 'k = 100')
 
subplot(1,2,2);
plot(recall_vector2(:,1,1), precision_vector2(:,1,1),'r',recall_vector2(:,1,2), precision_vector2(:,1,2),'b',recall_vector2(:,1,3), precision_vector2(:,1,3),'g')
 title('ROC for Regularized R and W')
 legend('k = 10', 'k = 50', 'k = 100')

figure;
subplot(3,1,1);
plot(recall_vector2(:,1,1,1), precision_vector2(:,1,1,1),'r',recall_vector2(:,1,1,2), precision_vector2(:,1,1,2),'b',recall_vector2(:,1,1,3), precision_vector2(:,1,1,3),'g')
 title('ROC for Regularized R and W for k=10')
 legend('lambda=0.01','lambda=0.1','lambda=1')
 
subplot(3,1,2); 
plot(recall_vector2(:,1,2,1), precision_vector2(:,1,2,1),'r',recall_vector2(:,1,2,2), precision_vector2(:,1,2,2),'b',recall_vector2(:,1,2,3), precision_vector2(:,1,2,3),'g')
 title('ROC for Regularized R and W for k=50')
 legend('lambda=0.01','lambda=0.1','lambda=1')
 
subplot(3,1,3); 
plot(recall_vector2(:,1,3,1), precision_vector2(:,1,3,1),'r',recall_vector2(:,1,3,2), precision_vector2(:,1,3,2),'b',recall_vector2(:,1,3,3), precision_vector2(:,1,3,3),'g')
 title('ROC for Regularized R and W for k=100')
 legend('lambda=0.01','lambda=0.1','lambda=1')
 
figure;
subplot(3,1,1);
plot(recall_vector3(:,1,1,1), precision_vector3(:,1,1,1),'r',recall_vector3(:,1,1,2), precision_vector3(:,1,1,2),'b',recall_vector3(:,1,1,3), precision_vector3(:,1,1,3),'g')
 title('ROC for Regularized swapped R and W for k=10')
 legend('lambda=0.01','lambda=0.1','lambda=1')
 
subplot(3,1,2); 
plot(recall_vector3(:,1,2,1), precision_vector3(:,1,2,1),'r',recall_vector3(:,1,2,2), precision_vector3(:,1,2,2),'b',recall_vector3(:,1,2,3), precision_vector3(:,1,2,3),'g')
 title('ROC for Regularized swapped R and W for k=50')
 legend('lambda=0.01','lambda=0.1','lambda=1')
 
subplot(3,1,3); 
plot(recall_vector3(:,1,3,1), precision_vector3(:,1,3,1),'r',recall_vector3(:,1,3,2), precision_vector3(:,1,3,2),'b',recall_vector3(:,1,3,3), precision_vector3(:,1,3,3),'g')
 title('ROC for Regularized swapped R and W for k=100')
 legend('lambda=0.01','lambda=0.1','lambda=1')
 