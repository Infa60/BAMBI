function [N_V] = f_t_norm(V)
% Compute the norm of the time vector V, with X,Y,Z in column and time in
% line, or quaternion q, with q1,q2,q3,q4 in column and time in line
type = size(V,2);
switch type
    case 2 % 2D Vector
        N_V=sqrt(V(:,1).*V(:,1)+V(:,2).*V(:,2));
    case 3 % 3D Vector
        N_V=sqrt(V(:,1).*V(:,1)+V(:,2).*V(:,2)+V(:,3).*V(:,3));
    case 4 % Quaternion
        N_V=sqrt(V(:,1).*V(:,1)+V(:,2).*V(:,2)+V(:,3).*V(:,3)+V(:,4).*V(:,4));
end
