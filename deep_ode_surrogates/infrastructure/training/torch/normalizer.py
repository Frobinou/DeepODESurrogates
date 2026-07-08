class TimeNormalizer:
    def __init__(self, t_min: float, t_max: float):
        self.t_min = t_min
        self.t_max = t_max

    def normalize(self, t):
        return 2.0 * (t - self.t_min) / (self.t_max - self.t_min) - 1.0

    def denormalize(self, tau):
        return (tau + 1.0) * (self.t_max - self.t_min) / 2.0 + self.t_min

    @property
    def dtau_dt(self):
        return 2.0 / (self.t_max - self.t_min)
