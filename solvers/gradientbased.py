import numpy as np


class GradientDescent:

    def __init__(self, gamma, maxit, tol, record=True):
        if isinstance(gamma, (float, int)):
            self.pace = lambda x: gamma
        elif callable(gamma):
            self.pace = gamma
        self.maxit = maxit
        self.tol = tol
        self.record = record

    def __call__(self, obj, grad, x0):
        it = 1
        sol = dict()
        gradeval = grad(x0)
        gradnorm = np.linalg.norm(gradeval)
        if self.record:
            gradnorms = [gradnorm]
            objs = [obj(x0)]
        x = x0.copy()
        while (it <= self.maxit) and (gradnorm > self.tol):
            x -= self.pace(it) * gradeval
            gradeval = grad(x)
            gradnorm = np.linalg.norm(gradeval)
            if self.record:
                gradnorms.append(gradnorm)
                objs.append(obj(x))
            it += 1
            print(it)
        if it == self.maxit:
            sol["success"] = False
        else:
            sol["success"] = True
        sol["x"] = x
        sol["func"] = obj(x)
        sol["gradnorm"] = gradnorm
        sol["nit"] = it
        if self.record:
            sol["gradnorm_record"] = gradnorms
            sol["func_record"] = objs
        return sol

