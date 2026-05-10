import numpy as np

def ideal_motion_yint(start: np.ndarray, vel: np.ndarray, target: np.ndarray):
    """
    :param start: end effector position
    :param vel: velocity vector of stream
    :param target: velocity

    :return: position at the x-coord of target
    """
    t = (target[0] - start[0]) / vel[0]
    y = -4.905 * t * t + vel[1] * t + start[1]
    return np.array([target[0], float(y)])

def general_order2(a, b, c):
    def polynomial(start: np.ndarray, vel: np.ndarray, target: np.ndarray):
        t = (target[0] - start[0]) / vel[0]
        y = a * t * t + b * vel[1] * t + start[1] + c
        return np.array([target[0], float(y)])
    return polynomial