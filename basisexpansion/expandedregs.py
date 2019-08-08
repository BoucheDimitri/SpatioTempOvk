from sklearn.linear_model import Ridge, Lasso


class ExpandedRidge:

    def __init__(self, lamb, funcdict):
        self.lamb = lamb
        self.funcdict = funcdict
        self.w = None

    def fit(self, X, y):
        Z = self.funcdict.eval(X)
        ridge = Ridge(alpha=self.lamb, fit_intercept=False)
        ridge.fit(Z, y)
        self.w = ridge.coef_.flatten()

    def predict(self, X):
        Z = self.funcdict.eval(X)
        return Z.dot(self.w)


class ExpandedLasso:

    def __init__(self, lamb, funcdict):
        self.lamb = lamb
        self.funcdict = funcdict
        self.w = None

    def fit(self, X, y):
        Z = self.funcdict.eval(X)
        lasso = Lasso(alpha=self.lamb, fit_intercept=False, max_iter=5000)
        lasso.fit(Z, y)
        self.w = lasso.coef_.flatten()

    def predict(self, X):
        Z = self.funcdict.eval(X)
        return Z.dot(self.w)