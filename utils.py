import re

def clean_code_output(raw_output):
    """
    Clean the model output to fix formatting issues
    """
    # Remove everything before "### Output:"
    if "### Output:" in raw_output:
        output = raw_output.split("### Output:")[-1].strip()
    else:
        output = raw_output
    
    # Fix: First, normalize all broken backtick patterns
    # Replace ``python, `` python, ```python, etc. with a marker
    output = re.sub(r'`{2,}\s*python', '###CODE_START###', output, flags=re.IGNORECASE)
    output = re.sub(r'`{2,}', '###CODE_END###', output)
    
    # Extract code between markers
    if '###CODE_START###' in output and '###CODE_END###' in output:
        parts = output.split('###CODE_START###')
        if len(parts) > 1:
            code_part = parts[1].split('###CODE_END###')[0]
            code = code_part.strip()
            
            # Clean up common issues
            code = code.replace('``python', '').replace('``', '')
            
            # Remove test code and comments after main function
            lines = code.split('\n')
            
            # Find where test code starts (usually after empty line + comment/print)
            clean_lines = []
            found_main_function = False
            empty_line_count = 0
            
            for line in lines:
                stripped = line.strip()
                
                # Track function definition
                if stripped.startswith('def '):
                    found_main_function = True
                    clean_lines.append(line)
                    empty_line_count = 0
                    continue
                
                # If we're in a function and hit empty lines
                if found_main_function and not stripped:
                    empty_line_count += 1
                    if empty_line_count <= 1:  # Allow one empty line
                        clean_lines.append(line)
                    continue
                
                # Stop at test code
                if found_main_function and empty_line_count > 0:
                    if stripped.startswith('#') or stripped.startswith('print('):
                        break  # Stop here, don't include test code
                
                # Include function body
                if found_main_function:
                    clean_lines.append(line)
                    empty_line_count = 0
            
            code = '\n'.join(clean_lines).strip()
            return f"```python\n{code}\n```"
    
    # Fallback: try original regex method
    code_blocks = re.findall(r'```(?:python)?\s*(.*?)```', output, re.DOTALL)
    if code_blocks:
        code = code_blocks[0].strip()
        # Remove test code
        lines = code.split('\n')
        main_code = []
        for line in lines:
            if line.strip().startswith('print(') or line.strip().startswith('# Test'):
                break
            main_code.append(line)
        
        code = '\n'.join(main_code).strip()
        return f"```python\n{code}\n```"
    
    # Last resort
    return output


def extract_code_only(raw_output):
    """
    Extract only the Python code without markdown formatting
    """
    cleaned = clean_code_output(raw_output)
    
    # Remove markdown code block markers
    code = re.sub(r'```python\s*', '', cleaned)
    code = re.sub(r'```\s*$', '', code)
    code = re.sub(r'``python\s*', '', code)  # Handle broken format
    
    return code.strip()


def detect_hallucination(output):
    """Detect if output contains hallucinated content"""
    hallucination_patterns = [
        (r'Human:', 'conversational'),
        (r'Assistant:', 'conversational'),
        (r'\*\*Created Question\*\*:', 'qa_generation'),
        (r'### Instruction:.*### Instruction:', 'repeated_instruction'),
    ]
    
    for pattern, hall_type in hallucination_patterns:
        if re.search(pattern, output, re.IGNORECASE | re.DOTALL):
            return True, hall_type
    
    return False, None


def validate_code_syntax(code):
    """Check if Python code is syntactically valid"""
    try:
        compile(code, '<string>', 'exec')
        return True, None
    except SyntaxError as e:
        return False, str(e)