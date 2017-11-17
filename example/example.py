import numpy as np
import rattlesnake as rs
print(rs.__version__)

# load fake data
trainX = np.loadtxt("trainX.txt")
trainY = np.loadtxt("trainY.txt")
testX = np.loadtxt("testX.txt")
testY = np.loadtxt("testY.txt")
n, p_x = trainX.shape
p_y = trainY.shape[1]
print(trainX.shape, trainY.shape)
print(testY.shape, testY.shape)

# normalize
# I already normalized the X before saving the data. So we would just do one on the Y.
# Note: It is empirically okay without centering at 0. But scaling is necessary.
sd_Y = np.std(trainY, 0).reshape([1, p_y])
trainY = trainY / sd_Y
print(np.std(trainY, 0))

# Define a single rattlesnake:
#   The complexity is defined by p. P is more like number of knots in smoothing splines.
#   The larger the p, the more complex the model becomes.
#   You can do deeper layers, but Rattlesnake is meant to achieve the "deep layer" performance without being deep.
p = 5
layers = [p_x, p, p_y]

# There are several link functions: linear, logit, softmax, logit_softmax (for numerical stability).
# Choice of loss functions: MSE, MSPE, CE.
model = rs.Rattlesnake(layers, loss="MSE", link="identity")

# initialize:
#   Rattlesnake can be initialized without data. But due to nature of the non-convex loss surface,
#   initialize with data can not only speed up the training, but also lead to better convergence region.
model.initialize(X=trainX, Y=trainY)

# train:
#   training stops when the seeing [ abs(percentage change of residuals) < perc_tol ] in 3 consecutive iterations
model.train(X=trainX, Y=trainY, perc_tol=1e-6, print_interval=1000, max_itr=150000)

# prediction and test error:
Y_hat = model.predict(testX) * sd_Y
print(np.mean(np.square(testY - Y_hat)))


# now try built-in ensemble (multiple snakes), with 20 models
#   ensemble can smooth the hypothesis function, hence,
#   a very good empirical technique for reaching to a good solution in a non-convex surface
#
#   built-in ensemble is faster and more memory-efficient than doing it manually
model_snakes = rs.Rattlesnake(layers, loss="MSE", link="identity", snake_num=20)
model_snakes.initialize(X=trainX, Y=trainY)
model_snakes.train(X=trainX, Y=trainY, perc_tol=1e-6, print_interval=1000, max_itr=150000)

# prediction and test error:
Y_hat_snakes = model_snakes.predict(testX) * sd_Y
print(np.mean(np.square(testY - Y_hat_snakes)))

# by default, ensemble takes the average in prediction. you can change it to median. Not necessarily better though.
Y_hat_snakes = model_snakes.predict(testX, if_mean=False) * sd_Y
print(np.mean(np.square(testY - Y_hat_snakes)))
