import numpy as np
from ..basic import gamma, gammaRatio

class Constant:
    '''
    f(x) = C
    '''

    def __init__(self, C=1):
        self.C = C
    
    def evaluate(self, xq, v=0, a=0):
        return self.C * (xq - a)**(-v) / gamma(1 - v)
    
    def diffint(self, v, a=0):
        def f(x):
            return self.evaluate(x, v, a)
        return f
    
    

class Linear:
    '''
    f(x) = x - a
    '''

    def __init__(self, a):
        self.a = a
    
    def evaluate(self, xq, v=0):
        return (xq - self.a)**(1 - v) / gamma(2 - v)
    
    def diffint(self, v):
        def f(x):
            return self.evaluate(x, v)
        return f
    



class Power:
    '''
    f(x) = (x - a)^p
    '''

    def __init__(self, p, a):
        self.p = p
        self.a = a
    
    def evaluate(self, xq, v=0):
        return gammaRatio(self.p + 1, self.p + 1 - v) * (xq - self.a)**(self.p - v)
    
    def diffint(self, v):
        def f(x):
            return self.evaluate(x, v)
        return f


class Polynomial:
    '''
    f(x) = \sum_{j=0}^\infty c_j (x - a)^{p + j/n}
    '''

    def __init__(self, a, n, p, c_arr):
        self.a = a
        self.c_arr = c_arr
        self.n = n
        self.p = p
        self.num_terms = len(c_arr)
    
    def evaluate(self, xq, v=0):
        gamma_arr = np.zeros(self.num_terms)
        for i in range(self.num_terms):
            gamma_arr[i] = gammaRatio((self.p*self.n + i + self.n) // self.n, (self.p*self.n - v*self.n + i + self.n) / self.n)
        result = 0
        coeff = c_arr * gamma_arr
        for i in range(self.num_terms):
            result = result + coeff[i] * np.power(xq - a, self.p - v + i/self.n)
        return  
    
    def diffint(self, v):
        def f(x):
            return self.evaluate(x, v)
        return f


class Binomial:
    '''
    f(x) = (C - kx)^p
    '''

    def __init__(self, C, k, p):
        self.C = C
        self.k = k
    
    def evaluate(self, xq, v=0, a=0):
        pass
    # 未完待续