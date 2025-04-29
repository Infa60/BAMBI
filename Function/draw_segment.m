% Function to draw a segment between two markers

function draw_segment(marker1, marker2)
    % Calculate the average (mean) position of marker1 and marker2 in the x and y directions
    line([mean(marker1(:,1)) mean(marker2(:,1))], ...  % X coordinates
         [mean(marker1(:,2)) mean(marker2(:,2))], ...  % Y coordinates
         'Color', 'r', 'LineWidth', 4);  % Set the line color to red and the width to 4
end