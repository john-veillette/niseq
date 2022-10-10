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
        assert(self.__call__(0) == 0) # starts at zero
        assert(self.__call__(max_n) == alpha) # ends at alpha
        a = [self.__call__(n) for n in range(max_n)]
        assert(np.all(np.diff(np.array(a)) > 0)) # and monotonically increases

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
