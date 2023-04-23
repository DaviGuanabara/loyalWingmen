import time
import curses
import numpy as np

stdscr = curses.initscr()


def formatObs(obs, reward, action):
    string = "Reward:" + "{:.4f}".format(reward) + " \n"
    # string += "Action:" + '{:.4f}'.format(reward) + " \n"

    foo = ["{:.4f}".format(m) for m in obs]
    keywords = [
        ("drone_position", 3),
        ("drone_velocity", 3),
        ("cube_position", 3),
        ("cube_velocity", 3),
        ("direction", 3),
        ("distance", 1),
    ]

    index_foo = 0

    for keyword, quantity in keywords:
        values = ""
        for i in range(quantity):
            if i == quantity - 1:
                values += foo[index_foo + i] + "\n"
            else:
                values += foo[index_foo + i] + ", "

        index_foo = index_foo + quantity
        string += keyword + ": " + values

    return string


def log(string):
    stdscr = curses.initscr()
    stdscr.addstr(0, 0, string)
    stdscr.refresh()


def logObs(obs, reward=0, action=[0, 0, 0, 0]):
    log(formatObs(obs, reward, action))


def test1():
    for i in range(10):
        log(0.01 * i)
        time.sleep(1)


def test2():
    for i in range(10):
        obs = [
            -0.00129556,
            -0.00431003,
            0.04844949,
            -0.00387202,
            -0.01718206,
            0.02817137,
            -0.00085829,
            0.00280715,
            0.00780588,
            0.00226181,
            -0.00142475,
            0.0035699,
            0.00043727,
            0.00711718,
            -0.04064361,
            0.00412644,
        ]

        logObs(np.array(obs) * i)

        time.sleep(1)


# test2()
