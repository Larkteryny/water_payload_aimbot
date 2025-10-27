import numpy as np
import math

class RobotModel():
    def __init__(self):
        pass

    def calc(self, start_pos, target_pos, p):
        return np.array([0])

class RobotModel_1Revolute(RobotModel):
    """
    1DOF model of fixed arm0 from starting point, a revolute joint, fixed arm1, and fixed end effector orientation
    """
    def __init__(self, LINK0, LINK1L, NOZZLE_V, NOZZLE_A, traj):
        super().__init__()
        self.traj = traj
        self.LINK0 = LINK0
        self.LINK0_A = math.atan2(self.LINK0[1], self.LINK0[0])
        self.LINK1l = LINK1L
        self.NOZZLE_V = NOZZLE_V
        self.NOZZLE_A = NOZZLE_A

    def calc(self, start_pos: np.ndarray, target_pos: np.ndarray, beta: float):
        """
        :param start_pos: relative position of revolute joint (position of LINK0's end NOT its start)
        :param target_pos: relative position of target
        :param beta: angle of revolute joint to test at
        """
        link1a = self.LINK0_A + beta
        effector = start_pos + [self.LINK1l * math.cos(link1a), self.LINK1l * math.sin(link1a)]
        out_a = link1a + self.NOZZLE_A
        vel = self.NOZZLE_V * np.array([math.cos(out_a), math.sin(out_a)])

        return self.traj(effector, vel, target_pos)

    def revolute_to_prismatic(self, start_pos, beta, origin, axis):
        """
        simple way of using existing math and tagging on an additional processing step to convert for a fixed-axis linear actuator

        :param start_pos: relative position of revolute joint (position of LINK0's end NOT its start)
        :param beta: angle of revolute joint
        :param origin: relative reference point of prismatic actuator
        :param axis: axis vector of the prismatic actuator
        """

        arm1a = self.LINK0_A + beta
        arm1 = [self.LINK1l * math.cos(arm1a), self.LINK1l * math.sin(arm1a)]

        # Find linear intersection of link1 with axis
        A = np.column_stack((arm1, -axis))
        b = origin - start_pos

        try:
            t1 = np.linalg.solve(A, b)
            p = start_pos + t1 * arm1
            if(p[0] < min(start_pos[0], start_pos[0] + arm1[0]) or p[0] > max(start_pos[0], start_pos[0] + arm1[0]) or
                p[1] < min(start_pos[1], start_pos[0] + arm1[1]) or p[1] > max(start_pos[1], start_pos[0] + arm1[1])):
                return None

            direction = -1 if any(x < 0 for x in np.divide((p - origin), axis)) else 1 # should be inf if /0
            return direction * np.linalg.norm(p - origin)
        except np.linalg.LinAlgError:
            return None

    # TODO: hinged linear actuator, only origin and length matter