function [u] = get_u(x1,x2,ref)
K = [0.1,0.1];
u0 = [0,0];
res = u0-K*([x1,x2] - ref.');
u=res(2);
end
