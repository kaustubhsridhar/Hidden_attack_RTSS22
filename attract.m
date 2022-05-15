function [c,ceq] = attract(x)
drift = 0;
threshold = 5;
initial = [0,0];
c(1) = -x(2)-drift-threshold;
final = [0,0];
for i = 1:99
    c(i+1)=-x(i*2+1)+1/3*x(i*2+1)^3-x(i*2+2)+0.1*get_u(x(2*i-1),x(2*i), final)-x(i*2+2)-drift-threshold;
end
ceq = [];
end