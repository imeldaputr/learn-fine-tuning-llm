======================================================================
Testing Fine-tuned Model (CodeAlpaca dataset)
======================================================================
Model: Qwen/Qwen2.5-Coder-1.5B-Instruct
Fine-tuned on: 100 CodeAlpaca examples
Post-processing: ✅ ENABLED
Test cases: 5
======================================================================

[Test 1/5] Write a function to calculate factorial
Category: Recursion | Training data: Likely ✓
----------------------------------------------------------------------
"""
    ```python
    def fact(n):
        if n == 0:
            return 1 
    
    else :
        result = 1  
        
        for i in range(2, (n+1)):    
            # multiply the number by each integer from two up until that number.     
            result *=i      
            
        //returning final product of all integers between one and 'num' inclusive.
            return(result)
    ```
"""
----------------------------------------------------------------------
📊 Analysis:
  • Raw length: 604 chars
  • Cleaned length: 365 chars
  • Reduction: 239 chars removed
  • Syntax valid: ❌ No - unindent does not match any outer indentation level (<string>, line 5)
  • Hallucination: ✅ None detected


[Test 2/5] Write a function to check palindrome
Category: String Manipulation | Training data: Likely ✓
----------------------------------------------------------------------
"""
    ```python
    def is_palindrome(s):
        # Remove all non-alphanumeric characters and convert the string into lowercase.
        s = ''.join(c.lower() for c in s if (c.isalnum()))
        
        return s == reverse_string(s)
    ```
"""
----------------------------------------------------------------------
📊 Analysis:
  • Raw length: 541 chars
  • Cleaned length: 213 chars
  • Reduction: 328 chars removed
  • Syntax valid: ✅ Yes
  • Hallucination: ✅ None detected


[Test 3/5] Create a function to find maximum in a list
Category: List Operations | Training data: Maybe ✓
----------------------------------------------------------------------
"""
    ```python
    def max_in_list(lst):
        if not lst:
            return None

        current_max = float('-inf')
        
        for num in lst:
        # Update the value of 'currentMax' only when we encounter 
            an element greater than it.
                    if (num > current_max):  
                    current_max=num
        
            return(current_max)
    ```
"""
----------------------------------------------------------------------
📊 Analysis:
  • Raw length: 575 chars
  • Cleaned length: 312 chars
  • Reduction: 263 chars removed
  • Syntax valid: ❌ No - inconsistent use of tabs and spaces in indentation (<string>, line 9)
  • Hallucination: ✅ None detected


[Test 4/5] Write a function to calculate the sum of even numbers in a list
Category: List + Conditionals | Training data: New ✗
----------------------------------------------------------------------
"""
    ```python
    def sum_even_numbers(lst):
        total = 0 
    
    for num in lst:
        if (num %2) == 0 :
            # add number into total variable.
                total += num
    
    return(total)
    ```
"""
----------------------------------------------------------------------
📊 Analysis:
  • Raw length: 554 chars
  • Cleaned length: 195 chars
  • Reduction: 359 chars removed
  • Syntax valid: ❌ No - unindent does not match any outer indentation level (<string>, line 4)
  • Hallucination: ✅ None detected


[Test 5/5] Create a function to merge two sorted lists
Category: Advanced Algorithm | Training data: New ✗
----------------------------------------------------------------------
"""
    ```python
    def MergeSortedLists(list1, list2):
        # Initialize an empty array for the merged result.
        MergedList = []
        
            # Iterate through both input arrays simultaneously until one of them is exhausted. 
            while len(List1) > 0 and len(LisT2)> 0:
            if List1[0] < LisT2 [0]:
                MergedList.append (list1.pop(0))
                
                else :
                        MergingList .append (lisT2.pop( 0 ))

    ## Add remaining elements from either lisst into merging_list   
            if(len(lisT1))>0 :  
            while(len(liST1)):     
                    MeringedList.append ((
    ```
"""
----------------------------------------------------------------------
📊 Analysis:
  • Raw length: 545 chars
  • Cleaned length: 549 chars
  • Reduction: -4 chars removed
  • Syntax valid: ❌ No - inconsistent use of tabs and spaces in indentation (<string>, line 6)
  • Hallucination: ✅ None detected


======================================================================
📊 SUMMARY REPORT
======================================================================

✅ Syntax Valid: 1/5 (20%)
⚠️  Hallucination: 0/5 (0%)
📏 Avg Raw Length: 564 chars
📏 Avg Cleaned Length: 327 chars
🧹 Total Cleaned: 1185 chars removed
📉 Reduction: 42.0%

======================================================================
Post-processing successfully cleaned all outputs! ✅
======================================================================