import numpy as np
import warnings
__version__ = "1.0"

#   all the warnings encountered so far have been checked,
#   they mainly exist as exp(-inf) which is okay.
#   if you want to see the warning, comment the line below
np.seterr(over="ignore", divide="ignore", invalid='ignore')

def __new_mat__(shape, val=1.0, dtype=np.float32):
    if val == 1.0:
        return np.ones(shape=shape, dtype=dtype)
    elif val == 0.0:
        return np.zeros(shape=shape, dtype=dtype)
    else:
        return np.multiply(np.ones(shape=shape, dtype=dtype), val)

class Rattlesnake(object):
    #   betas:      conventional weights parameter in neural network
    #   alphas:     linear/curvature parameter at each node; the larger, the stronger the linearity (simplicity)
    #   rates:      the magnitude of update on each individual parameter (see resilient back propagation 1993)
    #   etas:       scale of change in rates (see resilient back propagation 1993)
    #   steps:      store the previous parameter update. the signs are the ones mostly cared.
    #   loss:       specifying the loss function
    #   link:       specifying the link function
    #   layers:     layers of this rs object
    #   snake_num:  number of snakes in an ensemble
    #   beta_ave:   weight (non updatable) to compute the mean ensemble
    #   weights_avg: the weighting on the loss for each individual snake in an ensemble
    #   c:          a positive constraint scalar; indicating how strong the betas will be pulled
    #               towards correlation of +1 or -1. Burdens the computation.
    #               When in ensemble, it is not needed. When only using a single snake, it is recommended.
    #               -1.0 indicates adaptive constraint.
    #

    betas, rates_betas, steps_betas = None, None, None
    alphas, rates_alphas, steps_alphas = None, None, None
    etas, layers, loss, link = None, None, None, None
    dtype, maxexp = None, None
    snake_num, beta_avg, weights_avg = None, None, None
    c = None

    def __init__(self, layers, etas=None, loss="MSE", link="identity", dtype=np.float32, constraints=None, snake_num=1):
        if etas is None:
            etas = [0.5, 1.25]
        self.dtype = dtype
        self.layers = layers.copy()
        if snake_num > 1:
            self.snake_num = snake_num
            p = self.layers[-1]
            self.beta_avg = np.zeros((p * snake_num, p * (snake_num + 1)), dtype=self.dtype)
            temp = 1.0 / snake_num * np.eye(p)
            for i in range(snake_num):
                self.beta_avg[(i*p):(i*p+p), :p] = temp
            self.beta_avg[:, p:] = np.eye(snake_num * p)
            for i in range(1, len(self.layers)):
                self.layers[i] *= snake_num
            self.weights_avg = np.zeros((1, p * (snake_num + 1)), dtype=self.dtype)
            self.weights_avg[:, :p] = 0.5
            self.weights_avg[:, p:] = 0.5 / self.snake_num
        else:
            self.snake_num = 1
        self.etas = etas.copy()
        self.loss = loss
        self.link = link
        self.maxexp = np.finfo(dtype).maxexp / 2.0
        rate = 1.0
        self.betas = self.__new_mat_group__(self.layers, mode="betas", val=0.0)
        self.rates_betas = self.__new_mat_group__(self.layers, mode="betas", val=rate)
        self.steps_betas = self.__new_mat_group__(self.layers, mode="betas", val=0.0)
        self.alphas = self.__new_mat_group__(self.layers, mode="alphas", val=0.0)
        self.rates_alphas = self.__new_mat_group__(self.layers, mode="alphas", val=rate)
        self.steps_alphas = self.__new_mat_group__(self.layers, mode="alphas", val=0.0)
        if snake_num > 1:
            for i in range(1, len(self.betas)):
                p0 = int((self.betas[i].shape[0] - 1.) / self.snake_num)
                p1 = int(self.betas[i].shape[1] / self.snake_num)
                self.rates_betas[i][:, :] = 0.0
                self.rates_betas[i][0, :] = 1.0
                for j in range(snake_num):
                    self.rates_betas[i][(1+p0*j):(1+p0*j+p0), (j*p1):(j*p1+p1)] = 1.0

        if constraints is not None:
            if len(constraints) != len(self.betas):
                warnings.warn("constraints length is not consistent with the layers")
            for i in range(len(constraints)):
                if constraints[i] is not None:
                    a, b = constraints[i].shape
                    c, d = self.betas[i].shape
                    if a != c or b != d:
                        warnings.warn("constraint " + str(i) + " dimension incorrect")
                    self.rates_betas[i] = np.multiply(self.rates_betas, constraints[i])

    def __new_mat_group__(self, layers, mode="betas", val=0.0):
        mats = []
        if mode == "betas":
            for i in range(len(layers) - 1):
                mats.append(__new_mat__(shape=(layers[i] + 1, layers[i + 1]), val=val, dtype=self.dtype))
        elif mode == "alphas":
            for i in range(len(layers) - 2):
                mats.append(__new_mat__(shape=(layers[i + 1]), val=val, dtype=self.dtype))
        return mats

    def initialize(self, mode="random_normal", X=None, Y=None, r=None, rate=None, c=None, rho=0.50):
        if_normalization = X is not None
        if rate is None:
            rate = 0.1
            if self.loss == "CE":
                rate *= 0.05
        if r is None:
            radii = np.zeros(len(self.betas))
            for i in range(len(self.betas)):
                radii[i] = 1.0 * self.snake_num / self.betas[i].shape[1]
        elif type(r) == float or len(r) == 1:
            radii = np.zeros(len(self.betas))
            for i in range(len(self.betas)):
                radii[i] = r
        if c is None:
            self.c = 0.0
        for i in range(len(self.betas)):
            ind = np.not_equal(self.rates_betas[i], 0.0)
            self.rates_betas[i][ind] = 1.0

        ones, Z = None, None
        if mode == "random_normal":
            if if_normalization:
                n = X.shape[0]
                ones = np.ones((n, 1), dtype=self.dtype)
                Z = np.append(ones, X, axis=1)
            for i in range(len(self.betas)):
                p1, p2 = self.betas[i].shape
                rhos = (np.sqrt(np.repeat(rho, p2)) * np.random.choice([-1., 1.], p2)).reshape((p2, 1))
                S = (np.ones((p2, p2)) * rhos * rhos.T + (1 - rho) * np.eye(p2)) * (radii[i] * radii[i])
                self.betas[i] = np.random.multivariate_normal(np.repeat(0., p2), S, p1) * self.rates_betas[i]

                if if_normalization:
                    tempZ = np.matmul(Z, self.betas[i])
                    temp_std = np.std(tempZ, axis=0).reshape((1, self.betas[i].shape[1]))
                    temp_std[np.abs(temp_std) < 1e-6] = 1.0

                    self.betas[i] = self.betas[i] / temp_std
                    if i < len(self.betas) - 1:
                        Z, _, _ = self.__get_spline__(np.matmul(Z, self.betas[i]), self.alphas[i])
                        Z = np.append(ones, Z, axis=1)

        for i in range(len(self.alphas)):
            ind = np.not_equal(self.rates_alphas[i], 0.0)
            self.rates_alphas[i][ind] = rate / 3.0
            self.steps_alphas[i][ind] = 0.0
            self.alphas[i][:] = 1.0
        for i in range(len(self.betas)):
            self.steps_betas[i][:, :] = 0.0
            self.rates_betas[i] = self.rates_betas[i] * rate

    def train(self, X, Y, max_itr=10, perc_tol=1e-6, loss_tol=np.inf, step_tol=np.inf, zero_tol=0.0, print_interval=100,
              c=None, test_X=None, test_Y=None, test_loss=None, etas=None, perc_count_max=3):
        if_test = test_X is not None and test_Y is not None and test_loss is not None
        if c is not None:
            self.c = c
        if etas is not None:
            self.etas = etas
        if self.snake_num > 1:
            Y = np.tile(Y, self.snake_num+1)
        Z, dZdX, dZdAlphas = self.__forward_pass__(X)
        loss, dLoss = self.__get_loss__(Z[-1], Y)
        if self.snake_num > 1:
            loss_num = np.mean(loss * self.weights_avg)
        else:
            loss_num = np.mean(loss)
        loss = None
        perc_count = 0
        perc = None
        max_step_size = -np.inf

        itr = 0
        loss_tracker = np.array(np.repeat(np.nan, 4), dtype=self.dtype)
        while itr < max_itr and (perc_count < perc_count_max or loss_num > loss_tol) and max_step_size < step_tol:
            dBetas, dAlphas = self.__get_derivatives__(dLoss, Z, dZdX, dZdAlphas, loss_num, perc)
            dLoss, Z, dZdX, dZdAlphas = None, None, None, None
            max_step_size = -np.inf

            for i in range(0, len(dBetas)):
                temp_sign = np.sign(dBetas[i])
                if zero_tol > 0.0:
                    temp_sign[np.abs(dBetas[i]) < zero_tol] = 0.0
                self.rates_betas[i][np.equal(temp_sign, 0.0)] = 0.0
                if_same_sign = np.logical_and(np.equal(temp_sign, np.sign(self.steps_betas[i])), np.not_equal(temp_sign, 0.0))
                self.rates_betas[i] *= self.etas[0]
                self.rates_betas[i][if_same_sign] *= self.etas[1] / self.etas[0]
                if zero_tol > 0.0:
                    self.rates_betas[i][temp_sign == 0.0] = 0.0
                self.steps_betas[i] = np.multiply(temp_sign, self.rates_betas[i])
                if self.loss == "CE":
                    temp = self.betas[i].copy()
                    self.betas[i] = self.betas[i] - self.steps_betas[i]
                    # max_beta = 2.1972
                    max_beta = 2.9445
                    # max_beta = 5.0
                    self.betas[i][self.betas[i] > max_beta] = max_beta
                    self.betas[i][self.betas[i] < -max_beta] = -max_beta
                    self.steps_betas[i] = temp - self.betas[i]
                else:
                    self.betas[i] = self.betas[i] - self.steps_betas[i]
                max_step_size = np.max([max_step_size, np.max(np.abs(self.steps_betas[i]))])

            for i in range(0, len(dAlphas)):
                temp_sign = np.sign(dAlphas[i])
                if zero_tol > 0.0:
                    temp_sign[np.abs(dAlphas[i]) < zero_tol] = 0.0
                self.rates_alphas[i][np.equal(temp_sign, 0.0)] = 0.0
                if_same_sign = np.logical_and(np.equal(temp_sign, np.sign(self.steps_alphas[i])), np.not_equal(temp_sign, 0.0))
                self.rates_alphas[i] *= self.etas[0]
                self.rates_alphas[i][if_same_sign] *= self.etas[1] / self.etas[0]
                if self.loss == "CE":
                    self.rates_alphas[i][self.rates_alphas[i] > 0.01] = 0.01
                if zero_tol > 0.0:
                    self.rates_alphas[i][temp_sign == 0.0] = 0.0
                self.steps_alphas[i] = np.multiply(temp_sign, self.rates_alphas[i])
                if self.loss == "CE":
                    temp = self.alphas[i].copy()
                    self.alphas[i] = self.alphas[i] - self.steps_alphas[i]
                    self.alphas[i][self.alphas[i] > 100.0] = 100.0
                    self.alphas[i][self.alphas[i] < 0.0] = 0.0
                    self.steps_alphas[i] = temp - self.alphas[i]
                else:
                    self.alphas[i] = self.alphas[i] - self.steps_alphas[i]
                max_step_size = np.max([max_step_size, np.max(np.abs(self.steps_alphas[i]))])

            Z, dZdX, dZdAlphas = self.__forward_pass__(X)
            loss, dLoss = self.__get_loss__(Z[-1], Y)
            if self.snake_num > 1:
                new_loss_num = np.mean(loss * self.weights_avg)
            else:
                new_loss_num = np.mean(loss)
            loss = None
            perc = np.abs(new_loss_num / loss_num - 1.0)
            loss_num = new_loss_num
            if perc < perc_tol:
                perc_count += 1
                perc_count = np.min([perc_count, perc_count_max])
            else:
                perc_count = 0

            # loss track, check rate overshooting problem
            loss_tracker[1:] = loss_tracker[:(-1)]
            loss_tracker[0] = loss_num
            if itr > 4:
                # new minus old
                if np.mean(loss_tracker[:(-1)] - loss_tracker[1:] > 0.0) > 0.4:
                    for i in range(len(self.rates_alphas)):
                        self.rates_alphas[i] *= 0.5
                    for i in range(len(self.rates_betas)):
                        self.rates_betas[i] *= 0.5

            # print
            if (itr+1) % print_interval == 0:
                text = "iteration " + str(itr + 1) + "   " + str(self.loss) + " " + str(loss_num) + \
                       ",   perc "+str(perc) + "   max_step " + str(max_step_size)
                if if_test:
                    test_Z, _, _ = self.__forward_pass__(test_X)
                    test_Y_hat = test_Z[-1]
                    text = text + "   test error " + str(test_loss(test_Y, test_Y_hat))
                print(text)
            itr += 1
        itr -= 1
        if (itr+1) % print_interval != 0:
            text = "iteration " + str(itr + 1) + "   " + str(self.loss) + " " + str(loss_num) + \
                   ",   perc " + str(perc) + "   max_step " + str(max_step_size)
            if if_test:
                test_Z, _, _ = self.__forward_pass__(test_X)
                test_Y_hat = test_Z[-1]
                text = text + "   test error " + str(test_loss(test_Y, test_Y_hat))
            print(text)

    def __get_derivatives__(self, dLoss, Z, dZdX, dZdAlphas, loss_num, perc=None):
        n = Z[0].shape[0]
        dBetas = self.__new_mat_group__(self.layers, mode="betas", val=0.0)
        dAlphas = self.__new_mat_group__(self.layers, mode="alphas", val=0.0)
        if self.snake_num > 1:
            delta = np.multiply(np.matmul(dLoss, self.beta_avg.T), dZdX[-1])
        else:
            delta = np.multiply(dLoss, dZdX[-1])
        # delta[np.isnan(delta)] = 0.0

        for i in range(len(dBetas)):
            dBetas[i] = delta.copy()
        for i in range(len(dAlphas)):
            dAlphas[i] = delta.copy()
        for i in range(len(self.layers) - 2, -1, -1):
            dBetas[i] = np.matmul(Z[i].T, dBetas[i]) / (n * self.layers[-1])
            if i > 0:
                temp1 = np.matmul(dBetas[i - 1], self.betas[i][1:, :].T)
                temp2 = np.multiply(temp1, dZdX[i][:, 1:])
                for j in range(i-1, np.max([i-3, -1]), -1):
                    dAlphas[j] = temp1.copy()
                    dBetas[j] = temp2.copy()
            if i > 0:
                dAlphas[i - 1] = np.sum(np.multiply(dAlphas[i - 1], dZdAlphas[i - 1]), axis=0) / (n * self.layers[-1])

        if self.c != 0.0 and perc is not None:
            c = self.c
            if c < 0.0:
                c = loss_num
            if c > 0.0:
                for i in range(len(dBetas)-1):
                    if self.snake_num > 1:
                        p = int(dBetas[i].shape[1] / self.snake_num)
                        for j in range(self.snake_num):
                            dBetas[i][:, (j*p):(j*p+p)] = dBetas[i][:, (j*p):(j*p+p)] + c * self.__get_d_sum_cor_X_2__(self.betas[i][:, (j*p):(j*p+p)])
                    else:
                        dBetas[i] = dBetas[i] + c * self.__get_d_sum_cor_X_2__(self.betas[i])
        return dBetas, dAlphas

    def __forward_pass__(self, X):
        n = X.shape[0]
        ones = np.ones((n, 1), dtype=self.dtype)
        zeros = ones - 1.0
        Z, dZdX, dZdAlphas = [], [None], []

        Z.append(np.append(ones, X, axis=1))
        for i in range(1, len(self.layers) - 1):
            tempZ, tempZprimeX, tempZprimeAlpha = self.__get_spline__(np.matmul(Z[i - 1], self.betas[i - 1]), self.alphas[i - 1])
            Z.append(np.append(ones, tempZ, axis=1))
            dZdX.append(np.append(zeros, tempZprimeX, axis=1))
            dZdAlphas.append(tempZprimeAlpha)

        tempZ, tempZprime = self.__get_link__(np.matmul(Z[-1], self.betas[-1]))
        Z.append(tempZ)
        if self.snake_num > 1:
            Z.append(np.matmul(tempZ, self.beta_avg))
        dZdX.append(tempZprime)
        return Z, dZdX, dZdAlphas

    def __get_spline__(self, X, alpha, if_prime=True):
        A = np.matmul(np.ones((X.shape[0], 1)), alpha.reshape((1, X.shape[1])))
        g = 1.0 / (1.0 + np.exp(- A * X))
        if if_prime:
            prime = g * (1.0 - g)
            return g, A * prime, X * prime
        else:
            return g

    def __get_link__(self, X):
        if self.link == "identity":
            return X, np.ones(shape=X.shape, dtype=self.dtype)
        elif self.link == "logit":
            g = 1 / (1 + np.exp(-X))
            return g, np.multiply(g, 1 - g)
        elif self.link == "softmax":
            p = np.exp(X)
            row_sum = np.sum(p, 1).reshape((X.shape[0], 1))
            p = p / row_sum
            return p, p * (1.0 - p)
        elif self.link == "logit_softmax":
            g = 1.0 / (1.0 + np.exp(-X))
            row_sum = np.sum(g, axis=1).reshape((X.shape[0], 1))
            h = g / row_sum
            g_one_minus_g = g * (1.0 - g)
            h2 = g_one_minus_g / row_sum
            return h, (h2 - h * h2) * g_one_minus_g

    def __get_loss__(self, Y_hat, Y):
        if self.loss == "MSE":
            diff = Y_hat - Y
            loss = np.square(diff)
            dLoss = 2.0 * diff

        if self.loss == "MSPE":
            diff = (Y_hat - Y) / Y
            loss = np.square(diff)
            dLoss = 2.0 * diff / Y

        if self.loss == "CE":
            loss = - np.nan_to_num(Y * np.log(Y_hat)) - np.nan_to_num((1.0 - Y) * np.log(1.0 - Y_hat))
            dLoss = - np.nan_to_num(Y / Y_hat) + np.nan_to_num((1.0 - Y) / (1.0 - Y_hat))

        if self.snake_num > 1:
            dLoss = dLoss * self.weights_avg
        return loss, dLoss

    def __get_d_sum_cor_X_2__(self, X):
        n, p = X.shape
        mu_X = X - np.mean(X, 0).reshape((1, p))
        cov_X = np.matmul(mu_X.T, mu_X) / (n-1.)
        var_X = np.diag(cov_X).reshape((1, p))
        sd_X = np.sqrt(var_X)
        cor_X = cov_X / sd_X / sd_X.T

        A = 4. * cor_X / var_X / var_X.T / p / (p - 1.)
        B = A * sd_X * sd_X.T / (n-1.)
        I = np.eye(p)
        I_0 = 1. - I
        C = A * cov_X / sd_X / sd_X.T * var_X / (n-1.)

        return - np.matmul(mu_X, B*I_0) + np.matmul(mu_X, np.matmul(C, I_0) * I)

    def predict(self, X, if_mean=True):
        n = X.shape[0]
        ones = np.ones((n, 1), dtype=self.dtype)
        Z = np.append(ones, X, axis=1)

        for i in range(1, len(self.layers) - 1):
            tempZ = self.__get_spline__(np.matmul(Z, self.betas[i - 1]), self.alphas[i - 1], if_prime=False)
            Z = np.append(ones, tempZ, axis=1)

        tempZ, _ = self.__get_link__(np.matmul(Z, self.betas[-1]))
        if self.snake_num > 1:
            p = int(tempZ.shape[1]/self.snake_num)
            Y_hat = np.zeros((n, p))
            for j in range(p):
                temp = tempZ[:, np.linspace(0+j, p * self.snake_num - p+j, self.snake_num, dtype=int)]
                if if_mean:
                    Y_hat[:, j] = np.mean(temp, 1)
                else:
                    Y_hat[:, j] = np.median(temp, 1)
            if self.loss == "CE":
                row_sum = np.sum(Y_hat, 1).reshape((n, 1))
                Y_hat = Y_hat / row_sum
            return Y_hat
        else:
            return tempZ
