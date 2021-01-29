# adaptations to the gym_super_mario_bros environment found at https://github.com/Kautenja/gym-super-mario-bros
import gym_super_mario_bros as SMBGym


class SMBEnv(SMBGym.SuperMarioBrosEnv):
    def __init__(self):
        super().__init__()

    @staticmethod
    def make(game):
        return SMBGym.make(game)
