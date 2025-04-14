import re
from typing import Dict, List, Set, Optional, Union


def parse_steps(input_text: str) -> Dict[str, List[str]]:
    """
    Parses step output strings with different delimiters and formats.
    
    Args:
        input_text: The input string containing step tags
        
    Returns:
        A dictionary with step numbers as keys and lists of step types as values
    """
    # Initialize result dictionary
    result = {}
    
    # First, handle the complex case with detailed descriptions followed by OUTPUT: tags
    # This regex finds patterns like <step_2> text OUTPUT: <step_2> formal_reasoning | ... </step_2>
    complex_regex = r'<step_(\d+)>.*?OUTPUT:\s*<step_\1>\s*(.*?)\s*</step_\1>'
    complex_matches = re.finditer(complex_regex, input_text, re.DOTALL)
    
    for match in complex_matches:
        step_number = match.group(1)
        content = match.group(2).strip()
        
        # Parse the content based on different possible delimiters
        step_types = parse_step_content(content)
        result[step_number] = step_types
    
    # Then handle simpler cases without nested tags or with direct tags
    # Create a copy of the input text and remove already processed complex patterns
    input_text_simple = re.sub(complex_regex, '', input_text, flags=re.DOTALL)
    
    # This regex will match any remaining <step_X>content</step_X> patterns
    simple_regex = r'<step_(\d+)>((?!<step_).)*?</step_\1>'
    simple_matches = re.finditer(simple_regex, input_text_simple, re.DOTALL)
    
    for match in simple_matches:
        content = match.group(0).strip()
        step_number = match.group(1)
        
        # Skip if we already processed this step number from a complex match
        if step_number in result:
            continue
        
        # Extract the content between tags
        content_match = re.search(r'<step_\d+>(.*?)</step_\d+>', content, re.DOTALL)
        if content_match:
            content = content_match.group(1).strip()
        
        # If we have an OUTPUT: prefix, take only what's after it
        if 'OUTPUT:' in content:
            output_parts = content.split('OUTPUT:')
            content = output_parts[-1].strip()
        
        # Parse the content based on the format
        step_types = extract_step_types(content)
        result[step_number] = step_types
    
    return result


def extract_step_types(content: str) -> List[str]:
    """
    Extracts step types from content that may contain descriptions or explanations
    
    Args:
        content: The content to parse
        
    Returns:
        List of step types
    """
    # Check for colon format (e.g. "abduction: The model decides...")
    if ':' in content:
        # Split by newlines first to handle cases with multiple types
        lines = content.strip().split('\n')
        step_types = []
        
        for line in lines:
            line = line.strip()
            # If line contains a colon, extract the type
            if ':' in line:
                parts = line.split(':', 1)  # Split only on first colon
                type_part = parts[0].strip()
                step_types.extend(parse_step_content(type_part))
            # If line doesn't contain colon but has content, treat as type
            elif line:
                step_types.extend(parse_step_content(line))
        
        return step_types
    else:
        # Handle regular delimiter formats
        return parse_step_content(content)


def parse_step_content(content: str) -> List[str]:
    """
    Helper function to parse step content with different delimiters
    
    Args:
        content: The content to parse
        
    Returns:
        List of step types
    """
    step_types = []
    if 'classification:' in content:
        content = content.replace('classification:','')
    # Check for '|' delimiter
    if '|' in content:
        step_types = [item.strip() for item in content.split('|')]
    # Check for ';' delimiter
    elif ';' in content:
        step_types = [item.strip() for item in content.split(';')]
    # Check for ',' delimiter
    elif ',' in content:
        step_types = [item.strip() for item in content.split(',')]
    # Check for multiple words separated by spaces
    elif ' ' in content.strip() and len(content.strip().split()) > 1:
        step_types = content.strip().split()
    # Single item without delimiter
    else:
        step_types = [content.strip()]
    
    # Filter out any empty strings that might result from splitting
    return [item for item in step_types if item]


def get_all_unique_step_types(parsed_steps: Dict[str, List[str]]) -> List[str]:
    """
    Get all unique step types across all steps
    
    Args:
        parsed_steps: Dictionary of parsed steps
        
    Returns:
        List of all unique step types
    """
    unique_types = set()
    
    for step_types in parsed_steps.values():
        for step_type in step_types:
            unique_types.add(step_type)
    
    return list(unique_types)


def count_step_type_occurrences(parsed_steps: Dict[str, List[str]]) -> Dict[str, int]:
    """
    Count occurrences of each step type
    
    Args:
        parsed_steps: Dictionary of parsed steps
        
    Returns:
        Dictionary with step types as keys and occurrence counts as values
    """
    counts = {}
    
    for step_types in parsed_steps.values():
        for step_type in step_types:
            counts[step_type] = counts.get(step_type, 0) + 1
    
    return counts


def get_steps_with_type(parsed_steps: Dict[str, List[str]], target_type: str) -> Dict[str, List[str]]:
    """
    Get steps that contain a specific type
    
    Args:
        parsed_steps: Dictionary of parsed steps
        target_type: The step type to search for
        
    Returns:
        Dictionary with step numbers that contain the target type
    """
    matching_steps = {}
    
    for step_number, step_types in parsed_steps.items():
        if target_type in step_types:
            matching_steps[step_number] = step_types
    
    return matching_steps


# Example usage
if __name__ == "__main__":
    # Example input text
    example_text = """
    OUTPUT:
    <step_1> reading_problem | formal_reasoning </step_1>
    <step_2> formal_reasoning </step_2>
    <step_3> deduction </step_3>
    <step_4> deduction | formal_reasoning </step_4>
    <step_5> summarize, deduction </step_5>
    <step_6> finalized </step_6>
    """
    
    # Parse the steps
    parsed = parse_steps(example_text)
    print(parsed)
    
    # # Print the parsed results
    # import json
    # print(json.dumps(parsed, indent=2))
    
    # # Example of using the utility functions
    # print("\nAll unique step types:")
    # print(get_all_unique_step_types(parsed))
    
    # print("\nStep type occurrences:")
    # print(json.dumps(count_step_type_occurrences(parsed), indent=2))
    
    # print("\nSteps with 'formal_reasoning':")
    # print(json.dumps(get_steps_with_type(parsed, "formal_reasoning"), indent=2))