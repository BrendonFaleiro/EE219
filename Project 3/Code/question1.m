%%%%% Question 1 %%%%%%
filename = '..\Dataset\ml-100k\u.data';
delimiterIn = ('\t');
u= dlmread(filename, delimiterIn);
num = size(u,1);
%Creating an empty matrix R initialized with zeros
R = NaN(943,1682);
W = NaN(943,1682);
%Storing the values in R from the Dataset
for i=1:num
    R(u(i,1),u(i,2)) = u(i,3);
    W(u(i,1),u(i,2)) = 1;
end

k=[10, 50, 100];
e=zeros(1,3);
residual=zeros(1,3);

for a = 1:3
    [U,V,~,~,residual(a)] = wnmfrule(R,k(a));
    UV = U*V;
    for i = 1:943
        for j = 1:1682
            if isnan(R(i,j))==0
                e(a) = e(a) + (R(i,j)-UV(i,j))^2;
            end
        end
    end
    fprintf('Squared error = %.4f\n',e(a))
end
