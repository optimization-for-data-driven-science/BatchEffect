import numpy as np
import random

d = 1000
instances = 5000

mean = list(10 * np.random.rand(d))
A = np.random.rand(d, d)
cov = np.dot(A.T, A)

X = np.random.multivariate_normal(mean, cov, instances)
print(X)
print(X.shape)

theta = np.random.rand(d, 1)

for i in range(len(theta)):
    t = random.randint(1, 10)
    if t > 2:
        theta[i][0] = 0

noise = np.random.normal(0, 1, instances)
noise = noise[:, np.newaxis]

Y = np.dot(X, theta) + noise
full_data = np.concatenate((X, Y), axis=1)

np_params = []

X_train = X[0:1000]
Y_train = Y[0:1000]

X_test = X[1000:2000]
Y_test = Y[1000:2000]

print(X_train.shape)
print(Y_train.shape)

lam = 0
theta_star = np.dot(np.linalg.inv(np.dot(X_train.T, X_train) + lam * np.identity(1000)), np.dot(X_train.T, Y_train))

Y_pred = np.dot(X_test, theta_star)


nrmse = np.sqrt(np.linalg.norm(Y_test - Y_pred, 2)**2 / 1000) / np.std(Y_test)
print(nrmse)

print(random.random())
print(random.random())
print(random.random())
print(random.random())
print(random.random())
print(random.random())

for i in range(10):
    # Dataset i
    a_i = 10 * random.random()
    b_i = 10 * random.random()
    c_i = 10 * random.random()

    np_params.append([a_i, b_i, c_i])

    begin = int(i * instances / 10)
    end = int((i+1) * instances / 10)
    data_i = full_data[begin:end, :]

    name = "original_data_y_transformed" + str(i+1) + '.csv'
    np.savetxt(name, data_i, delimiter=",")
    # data_i = X[begin:end, :]
    # y_i = Y[begin:end, :]

    for j in range(data_i.shape[0]):
        for k in range(data_i.shape[1]):
            data_i[j][k] = a_i * data_i[j][k] * data_i[j][k] + b_i * data_i[j][k] + c_i + np.random.normal(0, 1, 1)[0]

    # full_data_i = np.concatenate((data_i, y_i), axis=1)

    # name = "data_y_not_transformed" + str(i+1) + '.csv'
    # np.savetxt(name, full_data_i, delimiter=",")

    name = "data_y_transformed" + str(i+1) + '.csv'
    np.savetxt(name, data_i, delimiter=",")

np.savetxt("theta_y_transformed.csv", theta, delimiter=",")

np.savetxt("params_y_transformed.csv", np.array(np_params))

