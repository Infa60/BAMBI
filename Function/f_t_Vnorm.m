function [normalised_V] = f_t_Vnorm(V)
norm = f_t_norm(V);
type = size(V,2);
switch type
    case 2
        normalised_V = V./[norm norm];
    case 3
        normalised_V = V./[norm norm norm];
    case 4
        normalised_V = V./[norm norm norm norm];
end
