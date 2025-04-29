function correctedMarkerFinal = f_correctMissingMarkers_rigid_body(markerData)
    numMarkers = size(markerData, 1);
    numFrames = size(markerData, 2);

    correctedMarkerFinal = markerData;

    for frame = 1:numFrames
        presentMarkers = ~isnan(markerData(:, frame)); % Marqueurs présents dans ce frame
        if sum(presentMarkers) >= 3 % Si au moins 3 marqueurs sont présents
            % Extraire les positions des marqueurs présents
            presentPositions = markerData(presentMarkers, frame);

            % Estimer la transformation rigide (rotation et translation)
            [R, t] = computeRigidBodyTransformation(presentPositions);

            % Prédire les positions des marqueurs manquants
            for marker = 1:numMarkers
                if isnan(markerData(marker, frame)) % Si le marqueur est manquant
                    predictedPosition = R * markerData(:, frame) + t;
                    correctedMarkerFinal(marker, frame) = predictedPosition(marker);
                end
            end
        end
    end
end

function [R, t] = computeRigidBodyTransformation(markerData)
    % Calcul de la transformation rigide (rotation et translation)
    % à partir des données des marqueurs disponibles

    % Sélectionner les données des marqueurs présents
    presentMarkers = ~isnan(markerData);
    presentPositions = markerData(presentMarkers, :);

    % Calculer le centroid des positions des marqueurs présents
    centroidInitial = mean(presentPositions, 2);

    % Centrer les données des marqueurs autour du centroid
    centeredMarkerData = presentPositions - centroidInitial;

    % Calculer la matrice de covariance croisée
    H = centeredMarkerData * centeredMarkerData';

    % Calculer la SVD de la matrice de covariance croisée
    [U, ~, V] = svd(H);

    % Calculer la rotation
    R = V * U';

    % Calculer la translation
    t = centroidInitial - R * centroidInitial;
end

