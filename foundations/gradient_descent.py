class Solution:
    def get_minimizer(self, iterations: int, learning_rate: float, init: int) -> float:
        for i in range(iterations):
            init -= learning_rate * (2 * init)
        return round(init, 5)