% Produit pour des matrices de dimensions L1*C1*n par des matrices de dimension L2*C2*n 
% n est le nombre d'images
% Utilisations dans le cadre des calculs de dynamique inverse
%___________________________________________________________________________
%
% Auteurs : Raphael DUMAS
% Date de création : Mars 2003
% Créé dans le cadre de : Post Doctorat
% Professeurs responsables :  J. de Guise et R. Assaoui
%___________________________________________________________________________
%
% Modifié par : Elsa NICOL
% Date de modification : Septembre 2005
% Créé dans le cadre de : Master Recherche
% Modification : Pour des matrices de dimension quelconque
%___________________________________________________________________________

function M3 = Mprod_array3 (M1,M2)

L1 = size (M1,1);
C1 = size (M1,2);
L2 = size (M2,1);
C2 = size (M2,2);
n = size (M1,3);

if L1==1==C1==1           %(M1 est un scalaire)
    for k = 1:n
        for i= 1:L2
            for j= 1:C2
                M3(i,j,k) = M1(1,1,k)*M2(i,j,k);
            end
        end
    end

elseif L2~=C1
    disp('dimensions des matrices incompatibles')
else
    % RAZ
    M3=[];
    % Transposée de M1
    M1t = permute (M1,[2,1,3]);
    for j=1:C2
        for i=1:L1
            % Produit terme à terme en dimension 1*1*n
            M3(i,j,:) = dot (M1t(:,i,:), M2(:,j,:));
        end
    end
end