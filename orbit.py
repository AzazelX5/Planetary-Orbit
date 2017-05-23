# coding=utf-8
from multiprocessing import Process, Queue
import time, math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import matplotlib.animation as animation

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
    ax1 = fig.add_subplot(2, 3, 4, xlim=(-2.5e5, 2.5e5), ylim=(-2.5e5, 2.5e5))
    ax2 = fig.add_subplot(5, 2, 8, xlim=(-2.5e5, 5e6), ylim=(-2.5e5, 2.5e5))
    ax3 = fig.add_subplot(4, 1, 1, xlim=(-2.5e5, 5e6), ylim=(-2.5e5, 2.5e5))

    line3_1_0, = ax3.plot([], [], '-', color='silver')
    line3_1_1, = ax3.plot([], [], marker='.', color='blue')

    line3_2_0, = ax3.plot([], [], '-', color='silver')
    line3_2_1, = ax3.plot([], [], marker='.', color='yellow')

    line3_3_0, = ax3.plot([], [], '-', color='silver')
    line3_3_1, = ax3.plot([], [], marker='.', color='cyan')

    line3_4_0, = ax3.plot([], [], '-', color='silver')
    line3_4_1, = ax3.plot([], [], marker='.', color='gold')

    line3_5_0, = ax3.plot([], [], '-', color='silver')
    line3_5_1, = ax3.plot([], [], marker='o', color='goldenrod')

    line3_6_0, = ax3.plot([], [], '-', color='silver')
    line3_6_1, = ax3.plot([], [], marker='o', color='tan')

    line3_7_0, = ax3.plot([], [], '-', color='silver')
    line3_7_1, = ax3.plot([], [], marker='o', color='aquamarine')

    line3_8_0, = ax3.plot([], [], '-', color='silver')
    line3_8_1, = ax3.plot([], [], marker='o', color='lightsteelblue')


    line1_0, = ax1.plot([], [], '-', color='silver')
    line1_1, = ax1.plot([], [], marker='.', color='blue')

    line2_0, = ax1.plot([], [], '-', color='silver')
    line2_1, = ax1.plot([], [], marker='.', color='yellow')

    line3_0, = ax1.plot([], [], '-', color='silver')
    line3_1, = ax1.plot([], [], marker='.', color='cyan')

    line4_0, = ax1.plot([], [], '-', color='silver')
    line4_1, = ax1.plot([], [], marker='.', color='gold')

    line5_0, = ax2.plot([], [], '-', color='silver')
    line5_1, = ax2.plot([], [], marker='o', color='goldenrod')

    line6_0, = ax2.plot([], [], '-', color='silver')
    line6_1, = ax2.plot([], [], marker='o', color='tan')

    line7_0, = ax2.plot([], [], '-', color='silver')
    line7_1, = ax2.plot([], [], marker='o', color='aquamarine')

    line8_0, = ax2.plot([], [], '-', color='silver')
    line8_1, = ax2.plot([], [], marker='o', color='lightsteelblue')


    cir1 = Circle(xy=(0.0, 0.0), radius=1e4, color='red')
    cir2 = Circle(xy=(0.0, 0.0), radius=3e4, color='red')
    cir3 = Circle(xy=(0.0, 0.0), radius=1e4, color='red')
    ax1.grid()
    ax2.grid()
    ax3.grid()

    x_1data, y_1data = [], []
    x_2data, y_2data = [], []
    x_3data, y_3data = [], []
    x_4data, y_4data = [], []
    x_5data, y_5data = [], []
    x_6data, y_6data = [], []
    x_7data, y_7data = [], []
    x_8data, y_8data = [], []

    def run(data):
        data_1, data_2, data_3, data_4, data_5, data_6, data_7, data_8 = data

        x_1data.append(data_1[0])
        y_1data.append(data_1[1])

        line1_0.set_data(x_1data, y_1data)
        line1_1.set_data(data_1[0],data_1[1])

        line3_1_0.set_data(x_1data, y_1data)
        line3_1_1.set_data(data_1[0], data_1[1])

        x_2data.append(data_2[0])
        y_2data.append(data_2[1])

        line2_0.set_data(x_2data, y_2data)
        line2_1.set_data(data_2[0], data_2[1])

        line3_2_0.set_data(x_2data, y_2data)
        line3_2_1.set_data(data_2[0], data_2[1])

        x_3data.append(data_3[0])
        y_3data.append(data_3[1])

        line3_0.set_data(x_3data, y_3data)
        line3_1.set_data(data_3[0], data_3[1])

        line3_3_0.set_data(x_3data, y_3data)
        line3_3_1.set_data(data_3[0], data_3[1])

        x_4data.append(data_4[0])
        y_4data.append(data_4[1])

        line4_0.set_data(x_4data, y_4data)
        line4_1.set_data(data_4[0], data_4[1])

        line3_4_0.set_data(x_4data, y_4data)
        line3_4_1.set_data(data_4[0], data_4[1])

        x_5data.append(data_5[0])
        y_5data.append(data_5[1])

        line5_0.set_data(x_5data, y_5data)
        line5_1.set_data(data_5[0], data_5[1])

        line3_5_0.set_data(x_5data, y_5data)
        line3_5_1.set_data(data_5[0], data_5[1])

        x_6data.append(data_6[0])
        y_6data.append(data_6[1])

        line6_0.set_data(x_6data, y_6data)
        line6_1.set_data(data_6[0], data_6[1])

        line3_6_0.set_data(x_6data, y_6data)
        line3_6_1.set_data(data_6[0], data_6[1])

        x_7data.append(data_7[0])
        y_7data.append(data_7[1])

        line7_0.set_data(x_7data, y_7data)
        line7_1.set_data(data_7[0], data_7[1])

        line3_7_0.set_data(x_7data, y_7data)
        line3_7_1.set_data(data_7[0], data_7[1])

        x_8data.append(data_8[0])
        y_8data.append(data_8[1])

        line8_0.set_data(x_8data, y_8data)
        line8_1.set_data(data_8[0], data_8[1])

        line3_8_0.set_data(x_8data, y_8data)
        line3_8_1.set_data(data_8[0], data_8[1])

        return line1_0, line2_0, line3_0, line4_0, line5_0, line6_0, line7_0, line8_0,\
               line1_1, line2_1, line3_1, line4_1, line5_1, line6_1, line7_1, line8_1,\
               line3_1_0, line3_2_0, line3_3_0, line3_4_0, line3_5_0, line3_6_0, line3_7_0, line3_8_0,\
               line3_1_1, line3_2_1, line3_3_1, line3_4_1, line3_5_1, line3_6_1, line3_7_1, line3_8_1



    ani= animation.FuncAnimation(fig, run, collect(q1, q2, q3, q4), blit=True, interval=1e-10, repeat=False)
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


