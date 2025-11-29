import re

def clean_code_output(raw_output):
    """
    Clean the model output to fix formatting issues
    
    Args:
        raw_output: Raw string output from model
        
    Returns:
        Cleaned output with proper formatting
    """
    # Remove everything before "### Output:"
    if "### Output:" in raw_output:
        output = raw_output.split("### Output:")[-1].strip()
    else:
        output = raw_output
        
    # Fix backtick issues
    # Replace excessive backticks (3+) with exactly 3
    output = re.sub(r'`{3,}', '```', output)
    
    # Extract only the first code block
    code_blocks = re.findall(r'```(?:python)?\s*(.*?)```', output, re.DOTALL)
    
    if code_blocks:
        # Take first complete code block
        clean_code = code_blocks[0].strip()
        
        # Format properly
        result = f"```python\n{clean_code}\n```"
        return result
    
    # If no code block found, try to extract code between backticks
    if '```' in output:
        # Take everything until first closing backticks
        parts = output.split('```')
        if len(parts) >= 2:
            code = parts[1].strip()
            # Remove 'python' tag if it's there
            if code.startswith('python'):
                code = code[6:].strip()
            return f"```python\n{code}\n```"
    
    # Fallback: return as-is
    return output


def extract_code_only(raw_output):
    """
    Extract only the Python code without markdown formatting
    
    Args:
        raw_output: Raw string output from model
        
    Returns:
        Pure Python code as string
    """
    cleaned = clean_code_output(raw_output)
    
    # Remove markdown code block markers
    code = re.sub(r'```python\s*', '', cleaned)
    code = re.sub(r'```\s*$', '', code)
    
    return code.strip()


def detect_hallucination(output):
    """
    Detect if output contains hallucinated content
    
    Args:
        output: Model output string
        
    Returns:
        tuple: (has_hallucination: bool, hallucination_type: str)
    """
    hallucination_patterns = [
        (r'Human:', 'conversational'),
        (r'Assistant:', 'conversational'),
        (r'\*\*Created Question\*\*:', 'qa_generation'),
        (r'### Instruction:.*### Instruction:', 'repeated_instruction'),
        (r'```python.*```python', 'repeated_code_block'),
    ]
    
    for pattern, hall_type in hallucination_patterns:
        if re.search(pattern, output, re.IGNORECASE | re.DOTALL):
            return True, hall_type
    
    return False, None


def validate_code_syntax(code):
    """
    Check if Python code is syntactically valid
    
    Args:
        code: Python code string
        
    Returns:
        tuple: (is_valid: bool, error_message: str)
    """
    try:
        compile(code, '<string>', 'exec')
        return True, None
    except SyntaxError as e:
        return False, str(e)