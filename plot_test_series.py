import pickle
import matplotlib.pyplot as plt
import numpy as np
import pertinence_funcs as pf
import partition_utilities as pu

series = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]

# Figure a

fig, ax = plt.subplots()
f_name = 'data_sets/series_1.pkl'
with open(f_name, 'rb') as f:
    t, y = pickle.load(f)

# Plot data
#ax.plot(t, y, '.')
#plt.ylim(15.5, 25)
#plt.xlim(-250, np.max(t)+250)

plt.ylim(5, 10)
plt.xlim(-0.2, 1.2)

# Plot lower and upper bounds
ub = np.max(y)
lb = np.min(y)

ref_line = np.linspace(0, np.max(t))

ub_line = np.ones(len(ref_line))*ub
lb_line = np.ones(len(ref_line))*lb

#ax.plot(ref_line, ub_line, color='black')
#ax.plot(ref_line, lb_line, color='black')

#ax.text(np.max(t), ub+0.1, r'$ub$', fontsize=11)
#ax.text(np.max(t), lb-0.5, r'$lb$', fontsize=11)

# Plot fuzzy sets

partitions = pu.generate_t_partitions(5, lb, ub)
#pf.plot_pertinence(partitions, 200)

print(partitions)
centers = [x[1] for x in partitions]

colors = ['orange', 'green', 'red', 'purple', 'brown']

count = 1
for c in centers:
    #ax.text(-120, c, r'$A_{}$'.format(count), fontsize=11, color=colors[count-1])
    count = count+1

# Set axis labels
#ax.set(xlabel='t', ylabel='Y(t)')

ax.set(xlabel=r'$m(x)$', ylabel='x')

plt.show()

fig.savefig('fig_triang', bbox_inches='tight')


# Figure b





# for s in series:
#     f_name = 'series_{}.pkl'.format(s)
#     #  Load series
#     with open(f_name, 'rb') as f:
#            t, y = pickle.load(f)
#
#     fig, ax = plt.subplots()
#     ax.plot(t, y)
#
#     ax.set(xlabel='time ', ylabel='y')
#
#     fig.savefig('series_{}.png'.format(s),bbox_inches='tight')


