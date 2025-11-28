# Comparison between the result of the fine-tuned and the original models

### Output Config
    max_new_tokens = 150 
    do_sample = False
    temperature = 0.7
    pad_token_id = tokenizer.eos_token_id

### Prompt
    "### Instruction: Write a function to calculate factorial\n### Output:"

## Result

## **The fine-tuned model**
    def factorial(n):
    if n == 0:
        return 1
    else:
        return n * factorial(n-1)

    def test_factorial():
        assert factorial(5) == 120, "Test case 1 failed"
        assert factorial(0) == 1, "Test case 2 failed"
        assert factorial(3) == 6, "Test case 3 failed"
        print("All test cases passed!")

    test_factorial()
    ```

    This code snippet defines a function `factorial` that calculates the factorial of a given number `n`. It uses recursion to compute the factorial. The `test_factorial` function is used to verify the correctness of the `factorial` function with


## **The original model**
    from typing import *
    from collections import *

    def factorial(n):
        if n == 0:
            return 1
        else:
            return n * factorial(n-1)

    print(factorial(5))  # Output: 120
    ```

    The provided code snippet defines a function `factorial` that calculates the factorial of a given number `n`. The function uses recursion to compute the factorial. If `n` is 0, it returns 1 (since the factorial of 0 is defined as 1). Otherwise, it returns `n` multiplied by the factorial of `n-1`.

    To test the function, we call `factorial(5)` and print the result, which


## Note/Observation
The fine-tuned model gives a more complete and useful answer (includes tests and broader explanation), but the formatting is messy and the explanation gets cut off.
The original model is cleaner and simpler, but less informative.