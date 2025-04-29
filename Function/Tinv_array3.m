function Tinv = Tinv_array3(M)

% Initialization
M1 = [];
M2 = [];

% First matrix
% Transposition of rotational part
% with transpose = permute ( , [2,1,3])
M1 (1:3,1:3,:) = permute (M(1:3,1:3,:),[2,1,3]);
M1 (4,4,:) = 1;

% Second matrix
% Opposite sign of the translation part
M2 (1:4,4,:) = - M (1:4,4,:);
M2 (1,1,:) = 1;
M2 (2,2,:) = 1;
M2 (3,3,:) = 1;
M2 (4,4,:) = 1;

% Matrix product
Tinv = Mprod_array3(M1,M2);



