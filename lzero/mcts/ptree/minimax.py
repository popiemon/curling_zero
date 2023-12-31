FLOAT_MAX = 1000000.0
FLOAT_MIN = -float('inf')


class MinMaxStats:

    def __init__(self) -> None:
        self.clear()
        self.value_delta_max = 0

    def set_delta(self, value_delta_max: float) -> None:
        self.value_delta_max = value_delta_max

    def update(self, value: float) -> None:
        if value > self.maximum:
            self.maximum = value
        if value < self.minimum:
            self.minimum = value

    def clear(self) -> None:
        self.minimum = FLOAT_MAX
        self.maximum = FLOAT_MIN

    def normalize(self, value: float) -> float:
        norm_value = value
        delta = self.maximum - self.minimum
        if delta > 0:
            if delta < self.value_delta_max:
                norm_value = (norm_value - self.minimum) / self.value_delta_max
            else:
                norm_value = (norm_value - self.minimum) / delta
        return norm_value


class MinMaxStatsList:

    def __init__(self, num: int) -> None:
        self.num = num
        self.stats_lst = [MinMaxStats() for _ in range(self.num)]

    def set_delta(self, value_delta_max: float) -> None:
        for i in range(self.num):
            self.stats_lst[i].set_delta(value_delta_max)
