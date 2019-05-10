import spatiotempovk.losses as losses
import spatiotempovk.regularizers as regularizers


class DiffSpatioTempRegressor:

    def __init__(self, loss, spacereg, timereg, mu, lamb):
        self.loss = loss
        self.spacereg = spacereg
        self.timereg = timereg
        self.mu = mu
        self.lamb = lamb

    #
    # def grad(self, Ms, y, Kx, Ks):
    #     def fixed_grad(alpha):
    #         g = objective_gradient(alpha.reshape((Ks.shape[0], Kx.shape[0])), loss_prime, Ms, y, Kx, Ks, mu, lamb)

