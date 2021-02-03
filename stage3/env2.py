# adaptations to the gym_super_mario_bros environment found at https://github.com/Kautenja/gym-super-mario-bros
import gym_super_mario_bros as SMBGym


class SMBEnv(SMBGym.SuperMarioBrosEnv):
    def __init__(self):
        super().__init__()
        self.prev_score = 0
        self.x_position_last = 0

    @staticmethod
    def make(game):
        return SMBGym.make(game)

    def _get_reward(self):
        existing_reward = self._time_penalty + self._death_penalty

        x_reward = self._x_position - self._x_position_last
        self.x_position_last = self._x_position

        if self._flag_get:
            flag_reward = 25
        else:
            flag_reward = 0

        score_reward = self._score + self.prev_score
        self.prev_score = self._score

        if self._player_state == 0x09:
            # agent cannot move - penalise
            state_reward = -5
        elif self._player_state in {0x07, 0x01, 0x02, 0x03}:
            # agent entering new area, descending pipe, or climbing vine - reward exploration
            state_reward = 5
        else:
            state_reward = 0

        return existing_reward + x_reward + flag_reward + score_reward + state_reward

    def _will_reset(self):
        # reset params to intial values when reset occurs
        self._time_last = 0
        self.x_position_last = 0
        self.prev_score = 0

    # def _x_reward(self):
        # currently sets reward equal to x position now - last x position (0 if >5 or <-5) then resets last x
        # x position is given by adding the current horizontal position (no. pixels from left) given by ram to previous accumulation
