import gym
import numpy as np
import universe
from gym.spaces import Box, Discrete
from universe import vectorized
from universe.wrappers import Unvectorize, Vectorize

import cv2


# Taken from https://github.com/openai/universe-starter-agent
def create_atari_env(env_id):
    env = gym.make(env_id)
    if len(env.observation_space.shape) > 1:
        env = Vectorize(env)
        env = AtariRescale42x42(env)
        env = NormalizedEnv(env)
        env = Unvectorize(env)
    return env


def create_car_racing_env():
    env = gym.make('CarRacing-v0')
    env = Vectorize(env)
    env = CarRacingRescale32x32(env)
    env = NormalizedEnv(env)
    env = CarRacingDiscreteActions(env)
    env = Unvectorize(env)
    return env


class CarRacingDiscreteActions(vectorized.ActionWrapper):

    def __init__(self, env=None):
        super(CarRacingDiscreteActions, self).__init__(env)
        self.action_space = Discrete(5)
        # 0 left
        # 1 right
        # 2 forward
        # 3 brake
        # 4 noop

    def _make_continuous_action(self, a):
        # print ("a = ", a)
        act = np.array([0., 0., 0.])
        if a == 0: # left
            act = np.array([-1., 0., 0.])
        elif a == 1: # right
            act = np.array([1., 0., 0.])
        elif a == 2: # gas
            act = np.array([0., 1., 0.])
        elif a == 3: # brake
            act = np.array([0., 0., 1.])
        elif a == 4: # noop
            act = np.array([0., 0., 0.])
        # print ("act: ", act)
        return act

    def _action(self, action_n):
        return [self._make_continuous_action(a) for a in action_n]

class CarRacingRescale32x32(vectorized.ObservationWrapper):

    def __init__(self, env=None):
        super(CarRacingRescale32x32, self).__init__(env)
        self.observation_space = Box(0.0, 1.0, [1, 32, 32])

    def _process_frame32(self, frame):
        frame = cv2.resize(frame, (32, 32))
        frame = frame.mean(2)
        frame = frame.astype(np.float32)
        frame *= (1.0 / 255.0)
        frame = np.reshape(frame, [1, 32, 32])
        return frame

    def _observation(self, observation_n):
        return [self._process_frame32(obs) for obs in observation_n]

def _process_frame42(frame):
    frame = frame[34:34 + 160, :160]
    # Resize by half, then down to 42x42 (essentially mipmapping). If
    # we resize directly we lose pixels that, when mapped to 42x42,
    # aren't close enough to the pixel boundary.
    frame = cv2.resize(frame, (80, 80))
    frame = cv2.resize(frame, (42, 42))
    frame = frame.mean(2)
    frame = frame.astype(np.float32)
    frame *= (1.0 / 255.0)
    frame = np.reshape(frame, [1, 42, 42])
    return frame


class AtariRescale42x42(vectorized.ObservationWrapper):

    def __init__(self, env=None):
        super(AtariRescale42x42, self).__init__(env)
        self.observation_space = Box(0.0, 1.0, [1, 42, 42])

    def _observation(self, observation_n):
        return [_process_frame42(observation) for observation in observation_n]


class NormalizedEnv(vectorized.ObservationWrapper):

    def __init__(self, env=None):
        super(NormalizedEnv, self).__init__(env)
        self.state_mean = 0
        self.state_std = 0
        self.alpha = 0.9999
        self.num_steps = 0

    def _observation(self, observation_n):
        for observation in observation_n:
            self.num_steps += 1
            self.state_mean = self.state_mean * self.alpha + \
                observation.mean() * (1 - self.alpha)
            self.state_std = self.state_std * self.alpha + \
                observation.std() * (1 - self.alpha)

        unbiased_mean = self.state_mean / (1 - pow(self.alpha, self.num_steps))
        unbiased_std = self.state_std / (1 - pow(self.alpha, self.num_steps))

        return [(observation - unbiased_mean) / (unbiased_std + 1e-8) for observation in observation_n]
