"""Utils for plotting the trajectory data
"""
import numpy as np
import matplotlib.pyplot as plt


def plot_positions(ax, poses, color='green'):
    """Plot the 3D trajectory positions
    """
    traj = []
    for pose in poses:
        traj.append(pose.t)
    traj = np.asarray(traj)
    assert(traj.shape == (len(poses), 3))
    ax.plot3D(traj[:, 0], traj[:, 1], traj[:, 2], color)


def plot_pose(ax, pose, length=1, style='-'):
    """Plot a single 3D pose
    """
    colors = ['r', 'g', 'b']
    lines = sum([ax.plot([], [], [], c=c) for c in colors], [])
    pos = pose.t
    start_points = np.array([pos, pos, pos])
    end_points = np.array([[length, 0, 0], [0, length, 0], [0, 0, length]]).T
    end_points = np.matmul(pose.R, end_points).T + start_points

    for line, start, end in zip(lines, start_points, end_points):
        line.set_data((start[0], end[0]), (start[1], end[1]))
        line.set_3d_properties((start[2], end[2]))
        line.set_linestyle(style)
        ax.draw_artist(line)


def plot_poses(ax, poses, length=1, style='-'):
    """Plot 3D trajectory poses
    """
    for pose in poses:
        plot_pose(ax, pose, length=length, style=style)


def plot_error(steps, errors):

    fontsize = 20

    plt.xticks(fontsize=fontsize)
    plt.yticks(fontsize=fontsize)

    plt.plot(steps, errors)
    plt.xlabel("step size", size=fontsize)
    plt.ylabel("error", size=fontsize)
    plt.show()


if __name__ == "__main__":
    steps = list(range(30, 0, -2))
    errors_1 = np.array([0.916042809626, 1.27206771899, 1.24252579758, 0.57626459169, 1.1011518421,
0.225221361446, 0.141257503239, 1.14023827121, 0.192965338033, 0.256529594336,
0.0941898260893, 0.0775161084819, 0.156229178107, 0.251318179103, 0.432600439038])

    errors_2 = np.array( [1.20350598185, 1.27992634968, 0.215328063695, 1.07739301562,
    1.00292204484, 0.151774748155, 0.209489328905, 1.23539909913, 0.157993574376,
    0.298702284554, 0.0899592738046, 0.110655748819, 0.121857887262,
    0.264863010456, 0.514414414957])

    errors_3 = np.array([0.7672405828914014, 1.2906270614007127, 0.2519317939358929, 0.5870292908792973, 1.1010814438239875, 0.2500712824357229, 0.16647481593550228, 1.2429984609174762, 0.2306516894859204, 0.30699335518995685, 0.12121165735148383, 0.10926757974505545, 0.11199048970374215, 0.1779311747754356, 0.5255828085448857]
)

    errors_4 = np.array([0.8440042483797955, 0.5823223761308773, 0.1746011127788541, 0.6253815531567742, 1.0583831889991955, 0.20248184393013496, 0.1209756002496279, 1.155650190161835, 0.15191929418503416, 0.32846448191443534, 0.19106480431394257, 0.09771144654506776, 0.17754438503290512, 0.21433812828340326, 0.46988337757945664])

    errors_5 = np.array([1.2833546104293245, 0.9799817248801148, 0.17785077281525385, 1.353132101800341, 0.7082490858484722, 0.19508426556318206, 0.24648737745006102, 1.2553413298919198, 0.2131050647513341, 0.30132896701522727, 0.1138327714148117, 0.11289358468457222, 0.14461774508843275, 0.1883223597180168, 0.5511388467860869])

    errors = (errors_1 + errors_2 + errors_3 + errors_4 + errors_5) / 5

    plot_error(steps[::-1], errors[::-1])