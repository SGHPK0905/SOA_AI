import numpy as np

def F1(x):
    """Hàm mục tiêu F1 (Sphere Function - Hàm đơn mode, hình cái bát)"""
    return np.sum(x**2)

def F2(x):
    """Hàm mục tiêu F2 (Schwefel 2.22 - Đơn mode)"""
    return np.sum(np.abs(x)) + np.prod(np.abs(x))

def F9(x):
    """Hàm mục tiêu F9 (Rastrigin Function - Hàm đa mode, rất nhiều bẫy)"""
    dimension = len(x)
    return np.sum(x**2 - 10 * np.cos(2 * np.pi * x)) + 10 * dimension

def F10(x):
    """Hàm mục tiêu F10 (Ackley - Đa mode, cực kỳ nhiều bẫy gai góc)"""
    dim = len(x)
    sum1 = np.sum(x**2)
    sum2 = np.sum(np.cos(2 * np.pi * x))
    term1 = -20 * np.exp(-0.2 * np.sqrt(sum1 / dim))
    term2 = -np.exp(sum2 / dim)
    return term1 + term2 + 20 + np.e

def fun_info(F):
    if F == 'F1':
        lowerbound = -100
        upperbound = 100
        dimension = 30
        fitness = F1
        return lowerbound, upperbound, dimension, fitness
    
    elif F == 'F2':
        lowerbound = -10
        upperbound = 10
        dimension = 30
        fitness = F2
        return lowerbound, upperbound, dimension, fitness
        
    elif F == 'F9':
        lowerbound = -5.12
        upperbound = 5.12
        dimension = 30
        fitness = F9
        return lowerbound, upperbound, dimension, fitness
    
    elif F == 'F10':
        lowerbound = -32.768
        upperbound = 32.768
        dimension = 30
        fitness = F10
        return lowerbound, upperbound, dimension, fitness
    
    else:
        print(f"Lỗi: Chưa định nghĩa hàm {F} trong hệ thống!")
        return None, None, None, None