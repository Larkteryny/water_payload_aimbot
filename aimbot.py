import numpy as np
import math

import robot_model as robot
import traj_model as traj


def _differentiate_err(model, start, target, p, i, r):
    at = np.linalg.norm(model(start, target, p[0]) - target)
    p[i] -= r
    below = np.linalg.norm(model(start, target, p[0]) - target)
    p[i] += 2*r
    above = np.linalg.norm(model(start, target, p[0]) - target)
    p[i] -= r
    dbelow = (below - at) / r
    dabove = (above - at) / r
    # TODO: more efficient sigmoid expressions
    magnitude = max(min(at / np.linalg.norm(target - start), 2), -2)
    return (magnitude * (2 / (1 + np.exp(-dbelow)) - 1), magnitude * (2 / (1 + np.exp(-dabove)) - 1))

def grad_desc(start: np.ndarray, target: np.ndarray, model: robot.RobotModel, p: list, scales: list, rate=0.001, max_iter = 5000):
    threshhold = 0.001
    derrs = [_differentiate_err(model.calc, start, target, p, i, scales[i]) for i in range(len(p))]

    iter_count = 0
    while(any(abs(de[0] - de[1]) > threshhold or de[0] < 0 or de[1] < 0 for de in derrs) and iter_count < max_iter):
        for i, de in enumerate(derrs):
            slope = de[0] - de[1]
            p[i] += slope * rate
        derrs = [_differentiate_err(model.calc, start, target, p, i, scales[i]) for i in range(len(p))]
        print(p, derrs)
        iter_count += 1
    print("Iterations: ", iter_count)
    return p

def newton_raphson():
    pass

if __name__ == "__main__":
    import turtle

    ARM0 = np.array((-0.1, -0.2))
    ARM1l = 0.3
    NOZZLE_V = 15
    NOZZLE_A = 10 / 180 * np.pi
    model = robot.RobotModel_1Revolute(ARM0, ARM1l, NOZZLE_V, NOZZLE_A, traj.ideal_motion_yint)

    start = np.array([0, 0])
    target = np.array([40, -20])
    p = [3*np.pi / 4]
    scales = [0.001]
    rate = 0.001
    grad_desc(start + ARM0, target, model, p, scales, rate)
    print(p[0])

    linear_origin = np.array([0.05, 0])
    linear_axis = np.array([-1, -2])
    distance = model.revolute_to_prismatic_fixed_axis(start+ARM0, p[0], linear_origin, linear_axis)
    print(distance)

    scalex = 500
    scaley = 500
    startd = start*[scalex, scaley]
    joint = startd + ARM0*[scalex, scaley]
    arm1a = model.LINK0_A + p[0]
    effectord = joint + [ARM1l * math.cos(arm1a), ARM1l * math.sin(arm1a)] * np.array([scalex, scaley])
    linear_origind = linear_origin*[scalex, scaley]
    linear_intersect = (linear_origin+distance*linear_axis/np.linalg.norm(linear_axis)) * [scalex, scaley]
    print(effectord * [1/scalex, 1/scaley])
    screen = turtle.Screen()
    screen.setup(width=700, height=700)
    main_turtle = turtle.Turtle()
    main_turtle.color("red")
    main_turtle.speed(0)
    main_turtle.teleport(float(scalex*target[0]), float(scaley*target[1]))
    main_turtle.dot(10)
    main_turtle.color("green")
    main_turtle.teleport(float(linear_origind[0]), float(linear_origind[1]))
    main_turtle.dot(5)
    main_turtle.goto(float(linear_intersect[0]), float(linear_intersect[1]))
    main_turtle.color("black")
    main_turtle.teleport(float(startd[0]), float(startd[1]))
    main_turtle.dot(5)
    main_turtle.goto(float(joint[0]), float(joint[1]))
    main_turtle.goto(float(effectord[0]), float(effectord[1]))
    main_turtle.color("blue")
    delta = (250 - effectord[0]) / scalex / 20
    for i in range(20):
        point = model.calc(start, np.array([3+i*delta, 0]), p[0]) * [scalex, scaley]
        main_turtle.goto(float(point[0]), float(point[1]))
    input()