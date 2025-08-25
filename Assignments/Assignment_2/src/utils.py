



def multiply(a,b):
    return a*b

def subtract(a,b):
    return a-b

def divide(a,b):
    if b==0:
        return "Division by zero is not allowed"
    return a/b

def factorial(n):
    if n==0:
        return 1
    else:
        return n * factorial(n-1)
    
    
