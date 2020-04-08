import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
import scipy.stats as stats

plt.rcParams['figure.figsize'] = [40, 20]

class FunktionAnlegen:
    def __init__(self, _range=10, res=.1, noise_max=2):
        self.range = _range
        self.noise_max = noise_max
        self.random_data = lambda x: x**2+np.random.rand()*noise_max
        self.data = np.array([[x*res for x in range(int(-_range/res), int(_range/res))],
                              [self.random_data(x*res) for x in range(int(-_range/res), int(_range/res))]])
        self.fkt = None
        self.test_set = None
        self.error_of_train_MSE = 0
        self.error_of_test_MSE = 0
        
    def plot_daten(self):
        plt.scatter(*self.data)
        
    def trainieren(self, n):
        if n >= 309:
            print("n is too big")
            raise ValueError
        self.fkt = np.poly1d(np.polyfit(*self.data, deg=n))
        
    def plot_fkt_ganzrat_fkt_n(self, n):     
        plt.scatter(*self.data)
        plt.plot(np.arange(-self.range, self.range, .001), [self.fkt(x) for x in np.arange(-self.range, self.range,.001)])
        plt.ylim([np.min(self.data)-10, np.max(self.data)+10])
    
    def plot_test_der_fkt(self, n, size):
        data_points = np.random.uniform(-self.range, self.range, size)
        self.test_set = np.array([data_points, [self.random_data(x) for x in data_points]])
        self.plot_fkt_ganzrat_fkt_n(n)
        plt.scatter(*self.test_set, c='yellow')
        plt.show()
        plt.subplot(211)
        plt.xlim(0,20)
        error_of_test = np.array([np.absolute(self.test_set[1][i]-self.fkt(x)) for i, x in enumerate(self.test_set[0])])
        self.error_of_test_MSE = np.square(error_of_test).mean()
        _ = plt.hist(np.clip(error_of_test,0,self.noise_max*2), int(size/2), (0,self.noise_max*2), density=True, color='yellow', label='Fehler Testdaten: {}'.format(str(self.error_of_test_MSE)))
        plt.legend(fontsize=20)
        plt.subplot(212)
        plt.xlim(0,20)
        error_of_train = np.array([np.absolute(self.data[1][i]-self.fkt(x)) for i, x in enumerate(self.data[0])])
        self.error_of_train_MSE = np.square(error_of_train).mean()
        _ = plt.hist(np.clip(error_of_train,0,self.noise_max*2), int(size/2), (0,self.noise_max*2), density=True, label='Fehler Trainingsdaten: {}'.format(str(self.error_of_train_MSE)))
        plt.legend(fontsize=20)

class NearestNeighbour:
    def __init__(self, k, n):
        self.k = k
        self.dots = np.array([np.array([np.random.rand(), np.random.rand()]) for _ in range(n)])
        self.color = lambda i:'blue' if i[0]*i[1]>.25 else 'red'
        self.c = np.array([self.color(i) for i in self.dots])
        self.distance = lambda i, j: np.sqrt(np.sum(np.array([a**2 for a in np.array(i-j)])))
        self.test_dot = None
        self.smallest_distance = None
        self.c_test = None
        self.nearest = None
        self.most_often = lambda arr: arr[np.argmax(np.unique(arr,return_counts=True)[1])]
    
    def plot_daten(self):
        plt.scatter([i[0] for i in self.dots], [i[1] for i in self.dots], c=self.c, s=150)
    
    def test(self):    
        self.test_dot = np.array([np.random.rand(), np.random.rand()])
        self.smallest_distance = [1e99 for _ in range(self.k)]
        self.c_test = [None for _ in range(self.k)]
        self.nearest = [None for _ in range(self.k)]
        for b, i in enumerate(self.dots):
            if self.distance(i,self.test_dot) < np.max(self.smallest_distance):
                j = self.smallest_distance.index(np.max(self.smallest_distance))
                self.c_test[j] = self.c[b]
                self.nearest[j] = b
                self.smallest_distance[j] = self.distance(i, self.test_dot)
        plt.scatter([i[0] for i in self.dots], [i[1] for i in self.dots], c=self.c, s=150)
        plt.scatter(*self.test_dot, c='yellow', s=200)
        for dot in self.nearest:
            plt.arrow(self.test_dot[0], self.test_dot[1], (self.dots[dot] - self.test_dot)[0], (self.dots[dot] - self.test_dot)[1])
        plt.text(0, 1.1, 'x: {}\ny: {}\n|vector|: {}\ncolor: {}'.format(str(self.test_dot[0]), str(self.test_dot[1]), self.smallest_distance, self.most_often(self.c_test)), fontsize=20)
