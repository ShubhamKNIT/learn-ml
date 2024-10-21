import pandas as pd
import matplotlib.pyplot as plt

# linear line (y = m * x + b) : loss function
def loss_function(m, b, points):
    total_error = 0
    for i in range(len(points)):
        x = points.iloc[i].studytime
        y = points.iloc[i].score
        total_error += (y - (m * x + b)) ** 2
    total_error / float(len(points))

# adjustment of m and b using gradient descent with defined lr
def gradient_descent(m_now, b_now, points, lr):
    m_gradient, b_gradient = 0, 0
    n = len(points)

    for i in range(n):
        x = points.iloc[i].studytime
        y = points.iloc[i].score

        m_gradient += -(2/n) * x * (y - (m_now * x + b_now))
        b_gradient += -(2/n) * (y - (m_now * x + b_now))
    
    m = m_now - m_gradient * lr
    b = b_now - b_gradient * lr
    return m, b

m, b  = 1, 0
lr = 0.01
epochs = 1000
data = pd.read_csv('data.csv')

for i in range(epochs):
    m, b = gradient_descent(m, b, data, lr)
    plt.scatter(data.studytime, data.score, color = 'red')
    plt.plot(range(0, 9), [m * x + b for x in range(0, 9)], color = 'blue')
    plt.pause(lr)
    if (i != epochs - 1):
        plt.clf()


print(m, b)
plt.show()