

class Schedule():

    def __init__(self, start_value, end_value):
        self.value = start_value
        self.end_value = end_value
        self.bound_function = max if start_value > end_value else min


class LinearSchedule(Schedule):
    """
    Schedule that is evolving as x(t+1) = x(t) + increment.
    """
    def __init__(self, start_value, end_value, steps):
        """
            start_value (float): initialisation value.
            end_value (float): final value.
            steps (int): number of steps to reach the end_value.
        """
        super().__init__(start_value, end_value)
        self.increment = (end_value - start_value) / steps

    def __call__(self):
        value = self.value
        self.value = self.bound_function(self.value + self.increment, self.end_value)
        return value


class ExponentialSchedule(Schedule):
    """
    Schedule that is evolving as x(t+1) = x(t) * factor.
    """
    def __init__(self, start_value, end_value, steps):
        """
            start_value (float): initialisation value.
            end_value (float): final value.
            steps (int): number of steps to reach the end_value.
        """
        super().__init__(start_value, end_value)
        self.factor = (end_value / start_value) ** (1 / steps)

    def __call__(self):
        value = self.value
        self.value = self.bound_function(self.value * self.factor, self.end_value)
        return value