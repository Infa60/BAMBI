% Function to check if the expected markers are present for a category

function category = check_category(markers, expected_markers)
    % Loop through each category in the expected_markers structure
    for cat = fieldnames(expected_markers)'
        category_name = cat{1};  % Get the category name (e.g., 'right', 'left')
        expected = expected_markers.(category_name);  % Get the expected markers for this category
        
        % Identify missing markers by comparing expected markers with actual markers
        missing_markers = setdiff(expected, markers.name);
        
        % If no markers are missing, assign this category
        if isempty(missing_markers)
            category = category_name;  % Return the category name if all markers are present
            return;
        end
    end
    
    % If no category is found (due to missing markers), return 'unknown'
    category = 'unknown';
end