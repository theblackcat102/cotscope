def get_top_classes(votings):
    """
    Analyze voting data and return classes with the largest normalized percentages for each step.
    If mathematics_operation is among the top classes, also include the second highest class.
    
    Args:
        votings (dict): Dictionary where keys are step numbers and values are lists of class votes
        
    Returns:
        dict: Dictionary where keys are step numbers and values are lists of classes with highest percentages
    """
    results = {}
    
    for step, votes in votings.items():
        # Count occurrences of each class
        counts = {}
        for vote in votes:
            counts[vote] = counts.get(vote, 0) + 1
        
        # Calculate normalized percentages
        total = len(votes)
        percentages = {class_name: count/total for class_name, count in counts.items()}
        
        # Find the maximum percentage
        max_percentage = max(percentages.values())
        
        # Find all classes with the maximum percentage
        max_classes = [class_name for class_name, percentage in percentages.items() 
                      if percentage == max_percentage]
        
        # Check if mathematics_operation is among the max classes
        if 'mathematics_operation' in max_classes:
            # Create a dictionary without the max classes
            remaining_percentages = {k: v for k, v in percentages.items() 
                                   if v != max_percentage}
            
            # If there are remaining classes, find the second highest
            if remaining_percentages:
                second_max = max(remaining_percentages.values())
                second_max_classes = [class_name for class_name, percentage in percentages.items() 
                                    if percentage == second_max]
                
                # Add the second highest classes to the result
                max_classes.extend(second_max_classes)
        
        # Store the list of top classes
        results[step] = max_classes
    
    return results
# Define the agreement calculation function
def calculate_agreement(pred1, pred2):
    """
    Calculate the agreement score between two model predictions.
    
    Args:
        pred1 (dict): First model's step predictions
        pred2 (dict): Second model's step predictions
        
    Returns:
        float: Agreement score between 0 and 1
    """
    # Get all steps from both predictions
    all_steps = set(pred1.keys()) | set(pred2.keys())
    
    if not all_steps:
        return 0.0  # No steps to compare
    
    agreements = 0
    
    for step in all_steps:
        # If a step is missing in one prediction, count as no agreement
        if step not in pred1 or step not in pred2:
            continue
        
        # Check for at least one common class between the predictions
        if set(pred1[step]) & set(pred2[step]):
            agreements += 1
    
    # Return the proportion of steps with agreement
    return agreements / len(all_steps)

# Optional: Generate a summary report
def generate_agreement_report(model_agreements):
    """Generate a summary of model agreements across all problems"""
    overall_agreements = {}
    
    # Initialize the overall agreement structure
    all_models = set()
    for problem in model_agreements:
        for model in model_agreements[problem]:
            all_models.add(model)
    
    for model1 in all_models:
        overall_agreements[model1] = {}
        for model2 in all_models:
            if model1 != model2:
                overall_agreements[model1][model2] = []
    
    # Collect all agreement scores
    for problem in model_agreements:
        for model1 in model_agreements[problem]:
            for model2 in model_agreements[problem][model1]:
                overall_agreements[model1][model2].append(model_agreements[problem][model1][model2])
    
    # Calculate average agreement scores
    average_agreements = {}
    for model1 in overall_agreements:
        average_agreements[model1] = {}
        for model2 in overall_agreements[model1]:
            scores = overall_agreements[model1][model2]
            if scores:
                average_agreements[model1][model2] = sum(scores) / len(scores)
            else:
                average_agreements[model1][model2] = 0.0
    
    return average_agreements