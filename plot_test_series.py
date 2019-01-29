import pickle
import matplotlib.pyplot as plt

series = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]

for s in series:
    f_name = 'series_{}.pkl'.format(s)
    #  Load series
    with open(f_name, 'rb') as f:
           t, y = pickle.load(f)

    fig, ax = plt.subplots()
    ax.plot(t, y)

    ax.set(xlabel='time ', ylabel='y')

    fig.savefig('series_{}.png'.format(s),bbox_inches='tight')


