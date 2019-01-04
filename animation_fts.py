from matplotlib import animation
import pertinence_funcs as pf
import numpy as np
import matplotlib.pyplot as plt
from fts_concrete import ConcreteFTS
from fts_stream import StreamAdaptiveWindowFTS

#Fts order
order = 3
nsets = 9
window_size = 100

# Gather sample data
n_samples = 100

x = np.linspace(-np.pi, 10*np.pi, n_samples)
data = 3 + np.sin(x)  # + np.random.normal(0, 0.1, len(x))
data2 = 3 + np.sin(x) + x

data = np.concatenate((data, data2, data+35))
t = np.arange(len(data))

lines = []

# First set up the figure, the axis, and the plot element we want to animate
fig = plt.figure()
ax = plt.axes(xlim=(-50, 310), ylim=(-5, 50))
line, = ax.plot([], [], lw=2)
line2, = ax.plot([], [], lw=2)
lines.append(line)
lines.append(line2)

ifts = StreamAdaptiveWindowFTS(nsets=nsets, order=order, bound_type='mu-sigma', update_type='retrain', deletion=True)


# initialization function: plot the background of each frame
def init():
    for i in range(nsets):
        nline, = ax.plot([], [], lw=1)
        lines.append(nline)

    for l in lines:
        l.set_data([], [])

    return lines

x = t[1:]
y = data[:-1]
ifts_forecast = []
partitions = []
samples_so_far = 0
count = 1
for d in data:
    print('{} of {}'.format(count, len(data)))
    count = count+1
    samples_so_far = samples_so_far + 1
    ifts_forecast.append(ifts.predict(d))
    if ifts.partitions:
        partitions.append(ifts.partitions)
    else:
        partitions.append([])

ts = np.arange(1, samples_so_far+1)


# animation function.  This is called sequentially
def animate(i):
    lines[0].set_data(x, y)
    lines[1].set_data(ts[:i], ifts_forecast[:i])

    if partitions[i]:
        dt = pf.plot_partitions_data(partitions[i])

        xs = dt[0]
        ys = dt[1]

        #lines[2].set_data(ys, xs)
        for i in range(nsets):
            lines[i+2].set_data(ys[i], xs[i])

    return tuple(lines)


# call the animator.  blit=True means only re-draw the parts that have changed.
anim = animation.FuncAnimation(fig, animate, init_func=init,
                               frames=len(x), interval=20, blit=True)

# save the animation as an mp4.  This requires ffmpeg or mencoder to be
# installed.  The extra_args ensure that the x264 codec is used, so that
# the video can be embedded in html5.  You may need to adjust this for
# your system: for more information, see
# http://matplotlib.sourceforge.net/api/animation_api.html
anim.save('incremental_fts.html', fps=30, extra_args=['-vcodec', 'libx264'])

plt.show()
