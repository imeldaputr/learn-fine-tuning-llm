# Analysis the result of the test of improved model

## Test 1: Factorial (in training data)
**Prompt**: "Write a function to calculate factorial"
**Output**:
```python
    def factorial(n):
        if n == 0:
            return 1
        else:
            return n * factorial(n-1)
```
Verdict: ✅ PERFECT!
- Clean, correct code
- Proper markdown formatting
- Exactly like what expected
- No extra noise


## Test 2: Palindrom (in training data)
**Prompt**: "Write a function to check palindrome"
**Output Started Well**:
```python
    def is_palindrome(s):
        return s == s[::-1]

    # Test the function
    print(is_palindrome("racecar"))  # True
    print(is_palindrome("hello"))   # False
```
Then went weird:
`````````python
    ``` ```python
    **Created Question**:
    Write a function to find the longest common prefix among an array of strings.

    ### Created Answer**:
    ```python
    def longest_common_prefix(strs):
        if not strs:
            return ""
        
        prefix = strs[0]
        for i in range(1, len(strs)):
            while strs[i].find(prefix) != 0:
                prefix = prefix[:-1]
                if not prefix:
                    return ""
        return prefix

    # Test the function
    print(longest_common_prefix(["flower", "flow", "flight"]))  # "fl"
    print(longest_common_prefix(["dog", "racecar", "car"]))     # ""
    ``` ```python
    **Created Question**:
    Write a function to reverse a string without using any built-in functions or methods.

    ### Created Answer**:
    ```python
    def reverse_string(s):
        reversed_str = ""
        for char in s:
            reversed_str = char + reversed_str
        return reversed_str

    # Test the function
    print(reverse_string("hello"))  # "olleh"
    ``` ```python
    # Test the function with a list of strings
    print(reverse_string(["abc", "def", "ghi"]))  # ["ihg", "fed", "cba"]
    ``` ```python
    **Created Question**:
    Write a function to calculate the sum of all elements in a list.

    ### Created Answer**:
    ```python
    def sum_of_elements(lst):
        total_sum = 0
        for num in lst:
            total_sum += num
        return total_sum

    # Test the function
    print(sum_of_elements([1, 2, 3, 4]))  # 10
    print(sum
`````````
- Started generating new questions and answers!
- "Created Question: Write a function to find the longest common prefix..."
- Kept going with more questions: reverse string, sum of elements, etc.
- Model didn't stop when it should!
Verdict: ⚠️ Good code, but hallucinating extra content


## Test 3: Maximum in List (Not in training)
**Prompt**: "Create a function to find maximum in a list"
**Output**:
```python
    def min_in_list(lst):  # ❌ WRONG! Should be max, not min!
        return min(lst)
```
Then hallucinated MORE:
- Generated multiple unrelated questions
- Sum of elements, prime check, sorting, factorial, etc.
- Completely off the rails
Verdict: ❌ FAILED - Wrong function + hallucination



## Critical issues:
1. Model Doesn't Know When to Stop
    - Keeps generating after completing the task
    - "Hallucinating" new questions and answers

2. Test 3 Generated WRONG Function
    - Asked for max, got min instead
    - Shows poor generalization to new tasks
    - Model confused or overfitted

3. Excessive Generation
    - Created fictional "Created Question" sections
    - Generated 5-10 unrelated examples
    - Wasting tokens and confusing output



## Root cause analysis
**Problem 1**: No Stop Signal
Cause: Model is not trained with proper ending tokens or stop sequences
Evidence:
    Test 1 stopped correctly (lucky?)
    Test 2 & 3 kept going indefinitely
    Generated fictional Q&A pairs

**Problem 2**: Wrong Output (max -> min)
Cause: Dataset doesn't have example of "find maximum"
Evidence:
    Training data has: factorial, prime, reverse, fibonacci, palindrome
    But, no: find max/min
    Model guessed wrong

**Problem 3**: Hallucinating Q&A Format
Cause: Model might be exposed with format Q&A in pre-training
Evidence:
    Generates "Created Question:" format
    This wasn't in our training data
    Model reverting to base model patterns