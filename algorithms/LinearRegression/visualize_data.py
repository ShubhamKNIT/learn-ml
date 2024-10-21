import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv('data.csv')

x = data['studytime']
y = data['score']

plt.scatter(x, y, c = 'red')
plt.show()