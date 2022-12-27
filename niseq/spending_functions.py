from abc import ABC, abstractmethod
import numpy as np

class SpendingFunction(ABC):
    '''
    An abstract base class for spending functions
    '''
    def __init__(self, alpha, max_n):
        # save specifications
        self.alpha = alpha
        self.max_n = max_n
        # and test that spending function is valid, a.k.a. it . . .
        assert(np.isclose(self.__call__(0), 0)) # starts at zero
        assert(np.isclose(self.__call__(max_n), alpha)) # ends at alpha
        a = [self.__call__(n) for n in range(max_n)]
        assert(np.all(np.diff(np.array(a)) >= 0)) # and monotonically increases

    @abstractmethod
    def __call__(self, n):
        pass

class LinearSpendingFunction(SpendingFunction):
    '''
    This is the simplest possible spending function, which
    distributes Type I error rate allowance evenly over time.
    '''
    def __call__(self, n):
        return n/self.max_n * self.alpha

class PocockSpendingFunction(SpendingFunction):
    '''
    A very common spending function that spends your alpha budget somewhat
    liberally (compared to e.g. the O'Brien-Fleming function) at the
    beginning of study; consequently, you have more power early on in
    exchange for a sharper penalty as you approach the maximum sample size.
    '''
    def __call__(self, n):
        return self.alpha * np.log(1 + (np.e - 1)*(n/self.max_n))

class OBrienFlemingSpendingFunction(SpendingFunction):
    '''
    A common choice for clinical trials or other confirmatory research, this
    spending function is conservative for early analyses, saving more power
    for later in the study.
    '''
    def __call__(self, n):
        from scipy.stats import norm
        r = n / self.max_n # information fraction
        if n != 0:
            return 2 * (1 - norm.cdf(norm.ppf(1 - self.alpha / 2) / np.sqrt(r)))
        else:
            return 0. # avoid divide by zero

class PiecewiseSpendingFunction(SpendingFunction):
    '''
    A piecewise spending funtion to be used when adjusting your maximum sample
    size. i.e., the old spending function `old_spending_func` is used up until
    `break_n`, the intermediate sample size at which you decided to change the
    max sample size. After that, a linear function is used that goes from
    old_spending_func(break_n) to (new_max_n, alpha).

    This is useful if, for instance, (1) you accidentally collect more data than
    your original max_n, requiring you to adjust your spending function, or
    (2) if a conditional power analysis encourages you to change your sample
    size to acheive a desired Type II error rate. Also (3) if you can no longer
    collect your original max_n for practical reasons.

    If max_n is adjusted multiple times, you can create piecewise spending
    functions recursively.
    '''

    def __init__(self, old_spending_func, break_n, new_max_n):
        self.old_spending_func = old_spending_func
        self.break_n = break_n
        super().__init__(old_spending_func.alpha, new_max_n)

    def __call__(self, n):
        if n <= self.break_n:
            return self.old_spending_func(n)
        else: # linear interpolation between spending at break point and max
            a = self.old_spending_func(self.break_n) # spending at break point
            m = (self.alpha - a)/(self.max_n - self.break_n)
            return m * (n - self.break_n) + a
