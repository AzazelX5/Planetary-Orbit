# coding=utf-8
from multiprocessing import Process, Queue
from matplotlib.patches import Circle
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import sys, math

mpl.rcParams[u'font.sans-serif'] = ['simhei']
mpl.rcParams['axes.unicode_minus'] = False

reload(sys)
sys.setdefaultencoding('utf-8')

def calculate(q, x_0, v_y_0, x_1, v_y_1):
    k1_x = k1_y = 0
    k2_x = k2_y = 0
    k3_x = k3_y = 0
    k4_x = k4_y = 0

    n = 0
    t = 0
    x = [[0.0, 0.0], [0.0, 0.0]]
    y = [[0.0, 0.0], [0.0, 0.0]]

    v_x = [[0.0, 0.0], [0.0, 0.0]]
    v_y = [[0.0, 0.0], [0.0, 0.0]]

    x[0][0] = x_0
    v_y[0][0] = v_y_0
    x[1][0] = x_1
    v_y[1][0] = v_y_1

    while True:
        G = 6.67 * (10 ** (-5))
        M = 1989000
        h = 2.0

        while t < 2:
            # 计算K1
            x0 = x[t][0]
            y0 = y[t][0]
            R = x0 * x0 + y0 * y0
            r = math.sqrt(R)
            a = G * M / R

            k1_x = -a * x0 / r
            k1_y = -a * y0 / r

            # 根据k1计算h/2时的数据
            v_x[t][1] = v_x[t][0] + 0.5 * h * k1_x
            v_y[t][1] = v_y[t][0] + 0.5 * h * k1_y

            x[t][1] = x[t][0] + v_x[t][1] * h * 0.5 + 0.25 * k1_x * h * h
            y[t][1] = y[t][0] + v_y[t][1] * h * 0.5 + 0.25 * k1_y * h * h

            # 计算k2
            x0 = x[t][1]
            y0 = y[t][1]
            R = x0 * x0 + y0 * y0
            r = math.sqrt(R)
            a = G * M / R

            k2_x = -a * x0 / r
            k2_y = -a * y0 / r

            # 根据k2计算出h/2时的数据
            v_x[t][1] = v_x[t][0] + 0.5 * h * k2_x
            v_y[t][1] = v_y[t][0] + 0.5 * h * k2_y

            x[t][1] = x[t][0] + v_x[t][1] * h * 0.5 + 0.25 * k2_x * h * h
            y[t][1] = y[t][0] + v_y[t][1] * h * 0.5 + 0.25 * k2_y * h * h

            # 计算k3
            x0 = x[t][1]
            y0 = y[t][1]
            R = x0 * x0 + y0 * y0
            r = math.sqrt(R)
            a = G * M / R

            k3_x = -a * x0 / r
            k3_y = -a * y0 / r

            # 根据k3计算出h时的数据
            v_x[t][1] = v_x[t][0] + 0.5 * h * k3_x
            v_y[t][1] = v_y[t][0] + 0.5 * h * k3_y

            x[t][1] = x[t][0] + v_x[t][1] * h + 0.5 * k3_x * h * h
            y[t][1] = y[t][0] + v_y[t][1] * h + 0.5 * k3_y * h * h

            # 计算k4
            x0 = x[t][1]
            y0 = y[t][1]
            R = x0 * x0 + y0 * y0
            r = math.sqrt(R)
            a = G * M / R

            k4_x = -a * x0 / r
            k4_y = -a * y0 / r

            v_x[t][1] = v_x[t][0] + h / 6 * (k1_x + 2 * k2_x + 2 * k3_x + k4_x)
            v_y[t][1] = v_y[t][0] + h / 6 * (k1_y + 2 * k2_y + 2 * k3_y + k4_y)

            x[t][0] += 0.5 * h * (v_x[t][1] + v_x[t][0])
            y[t][0] += 0.5 * h * (v_y[t][1] + v_y[t][0])

            v_x[t][0] = v_x[t][1]
            v_y[t][0] = v_y[t][1]

            n += 1
            t += 1

            if n > 20000:
                q.put([[x[0][0], y[0][0]], [x[1][0], y[1][0]]])
                n = 0

            if t == 2:
                t = 0

def collect(q1, q2, q3, q4):
    while True:
        n1 = q1.get(True)
        data_1_1 = n1[0]
        data_1_2 = n1[1]

        n2 = q2.get(True)
        data_2_1 = n2[0]
        data_2_2 = n2[1]

        n3 = q3.get(True)
        data_3_1 = n3[0]
        data_3_2 = n3[1]

        n4 = q4.get(True)
        data_4_1 = n4[0]
        data_4_2 = n4[1]

        yield data_1_1, data_1_2, data_2_1, data_2_2, data_3_1, data_3_2, data_4_1, data_4_2

def plot():
    fig = plt.figure(figsize = (20,10))
    ax1 = fig.add_subplot(4, 1, 1, xlim=(-2.5e5, 5e6), ylim=(-2.5e5, 2.5e5))
    ax2 = fig.add_subplot(2, 3, 4, xlim=(-2.5e5, 2.5e5), ylim=(-2.5e5, 2.5e5))
    ax3 = fig.add_subplot(5, 2, 8, xlim=(-2.5e5, 5e6), ylim=(-2.5e5, 2.5e5))
    ax4 = fig.add_subplot(6, 1, 3)

    ax4.set_xticks([])
    ax4.set_yticks([])
    ax4.spines['right'].set_color('none')
    ax4.spines['top'].set_color('none')
    ax4.spines['bottom'].set_color('none')
    ax4.spines['left'].set_color('none')


    list_track_1 = [['line1_0', 'line1_1'], ['line2_0', 'line2_1'], ['line3_0', 'line3_1'], ['line4_0', 'line4_1'],
                    ['line5_0', 'line5_1'], ['line6_0', 'line6_1'], ['line7_0', 'line7_1'], ['line8_0', 'line8_1']]
    list_color_1 = ['blue', 'yellow', 'cyan', 'gold', 'goldenrod', 'tan', 'aquamarine', 'lightsteelblue']

    list_track_2 = [['line1_0', 'line1_1'], ['line2_0', 'line2_1'], ['line3_0', 'line3_1'], ['line4_0', 'line4_1']]
    list_color_2 = ['blue', 'yellow', 'cyan', 'gold']

    list_track_3 = [['line5_0', 'line5_1'], ['line6_0', 'line6_1'], ['line7_0', 'line7_1'], ['line8_0', 'line8_1']]
    list_color_3 = ['goldenrod', 'tan', 'aquamarine', 'lightsteelblue']

    list_track_4 = ['line0', 'line1', 'line2', 'line3', 'line4', 'line5', 'line6', 'line7', 'line8']
    list_color_4 = ['red', 'blue', 'yellow', 'cyan', 'gold', 'goldenrod', 'tan', 'aquamarine', 'lightsteelblue']
    list_name_US = ['sun', 'Mercury', 'Venus', 'Earth', 'Mars', 'Jupiter', 'Saturn', 'Uranus', 'Neptune']
    list_name_CN = [u'太阳', u'水星', u'金星', u'地球', u'火星', u'木星', u'土星', u'天王星', u'海王星']

    n1 = n2 = n3 = n4 = 0
    x_data = [[46001.2], [107476.259], [152097.7], [206644.545], [740520.0], [1353572.956], [2748938.461],[4452940.833]]
    y_data = [[0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0]]

    for each_1 in list_track_1:
        each_1[0], = ax1.plot([], [], '-', color='silver')
        each_1[1], = ax1.plot([], [], 'o', color=list_color_1[n1])

        n1 += 1

    for each_2 in list_track_2:
        each_2[0], = ax2.plot([], [], '-', color='silver')
        each_2[1], = ax2.plot([], [], marker='o', color=list_color_2[n2])

        n2 += 1

    for each_3 in list_track_3:
        each_3[0], = ax3.plot([], [], '-', color='silver')
        each_3[1], = ax3.plot([], [], marker='o', color=list_color_3[n3])

        n3 += 1

    for each_4 in list_track_4:
        each_4, = ax4.plot([], [], 'o', color=list_color_4[n4])
        each_4.set_label(list_name_CN[n4])
        ax4.legend(loc = 9, ncol = 9)

        n4 += 1

    def run(data):
        t1 = t2  = 0
        t3 = 4

        for each_1 in list_track_1:
            x_data[t1].append(data[t1][0])
            y_data[t1].append(data[t1][1])
            each_1[0].set_data(x_data[t1], y_data[t1])
            each_1[1].set_data(data[t1][0], data[t1][1])

            t1 += 1

        for each_2 in list_track_2:
            each_2[0].set_data(x_data[t2], y_data[t2])
            each_2[1].set_data(data[t2][0], data[t2][1])

            t2 += 1

        for each_3 in list_track_3:
            each_3[0].set_data(x_data[t3], y_data[t3])
            each_3[1].set_data(data[t3][0], data[t3][1])

            t3 += 1

        return list_track_1[0][0], list_track_1[0][1], list_track_1[1][0], list_track_1[1][1], list_track_1[2][0], list_track_1[2][1], list_track_1[3][0], list_track_1[3][1], \
               list_track_1[4][0], list_track_1[4][1], list_track_1[5][0], list_track_1[5][1], list_track_1[6][0], list_track_1[6][1], list_track_1[7][0], list_track_1[7][1], \
               list_track_2[0][0], list_track_2[0][1], list_track_2[1][0], list_track_2[1][1], list_track_2[2][0], list_track_2[2][1], list_track_2[3][0], list_track_2[3][1], \
               list_track_3[0][0], list_track_3[0][1], list_track_3[1][0], list_track_3[1][1], list_track_3[2][0], list_track_3[2][1], list_track_3[3][0], list_track_3[3][1]

    cir1 = Circle(xy=(0.0, 0.0), radius=1e4, color='red')
    cir2 = Circle(xy=(0.0, 0.0), radius=1e4, color='red')
    cir3 = Circle(xy=(0.0, 0.0), radius=3e4, color='red')

    ax1.grid()
    ax2.grid()
    ax3.grid()

    ani = animation.FuncAnimation(fig, run, collect(q1, q2, q3, q4), blit=True, interval=1e-10, repeat=False)
    ax1.add_patch(cir1)
    ax2.add_patch(cir2)
    ax3.add_patch(cir3)
    plt.show()

q1 = Queue()
q2 = Queue()
q3 = Queue()
q4 = Queue()

t1 = Process(target=calculate, args=(q1, 46001.2, 0.053682, 107476.259, 0.035258))
t2 = Process(target=calculate, args=(q2, 152097.7, 0.029, 206644.545, 0.026499))
t3 = Process(target=calculate, args=(q3, 740520.0, 0.013719, 1353572.956, 0.010175))
t4 = Process(target=calculate, args=(q4, 2748938.461, 0.0071, 4452940.833, 0.0054899))
t5 = Process(target=collect, args=(q1, q2, q3, q4))
t6 = Process(target=plot, args=())

t1.start()
t2.start()
t3.start()
t4.start()
t5.start()
t6.start()

t1.join()
t2.join()
t3.join()
t4.join()
t5.join()
t6.join()

