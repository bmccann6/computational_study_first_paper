def calculate_expected_and_total_values_detected_this_prob_dist_across_resource_sets(prob_dist, resource_sets, hiding_locations, detectors):
    
    expected_value_each_node_this_prob_dist = calculate_expected_value_each_node_this_prob_dist(prob_dist, resource_sets, hiding_locations)
    optimal_list_detectors_this_prob_dist = get_optimal_list_detectors_this_prob_dist(budget, prob_dist, resource_sets, hiding_locations, detectors)
    
    
    # Convert dictionary to list and include accuracy for sorting
    solution_list = [(key.split('[')[0], detectors[key.split('[')[0]]['accuracy'], value) 
                       for key, value in optimal_list_detectors_this_prob_dist.items() if value > 0]

    # Sort by accuracy
    sorted_optimal_detectors_this_prob_dist = sorted(solution_list, key=lambda x: x[1], reverse=True)
 
    expected_value_detected_this_prob_dist = sum(sorted_optimal_detectors_this_prob_dist[i] * expected_value_each_node_this_prob_dist[i] for i in range(len(expected_value_each_node_this_prob_dist)))
    expected_total_value_this_prob_dist = sum(expected_value_each_node_this_prob_dist.values())
    
    return expected_value_detected_this_prob_dist, expected_total_value_this_prob_dist