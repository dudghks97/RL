import numpy as np
import random
from collections import defaultdict
from environment import Env


# 몬테카를로 에이전트 (모든 에피소드 각각의 샘플로 부터 학습)
class MCAgent:
    def __init__(self, actions):
        self.width = 7              # Grid World 너비
        self.height = 7             # Grid World 높이
        self.actions = actions      # 행동
        self.learning_rate = 0.01   # 학습률
        self.discount_factor = 0.9  # 감가율
        self.epsilon = 0.1          # 입실론
        self.samples = []           # sample 리스트
        self.value_table = defaultdict(float)                               # 가치함수 table
        self.percentage_table = defaultdict(lambda: [0.0, 0.0, 0.0, 0.0])   # 행동 확률 table
        self.reward_table = defaultdict(lambda: [0.0, 0.0, 0.0, 0.0])       # 보상 table

    # 메모리에 샘플을 추가
    def save_sample(self, state, reward, done):
        self.samples.append([state, reward, done])

    # 가능한 다음 상태들의 보상을 저장
    def save_reward(self, state, action, reward):
        state = str(state)
        self.reward_table[state][action] = reward

    # 모든 에피소드에서 에이전트가 방문한 상태의 큐 함수를 업데이트
    def update(self):
        visit_state = []
        sample_size = len(self.samples)
        for i, forward in enumerate(self.samples):
            state, reward, done = forward
            state = str(state)
            if state not in visit_state:                # first visit
                visit_state.append(state)
                G_t = 0                                 # MC Sample
                for j, reverse in enumerate(reversed(self.samples)):
                    if sample_size - j - 1 < i:
                        break
                    rev_state, rev_reward, rev_done = reverse
                    rev_state = str(rev_state)
                    G_t = rev_reward + self.discount_factor * G_t
                value = self.value_table[state]
                self.value_table[state] = (value + self.learning_rate * (G_t - value))

    # 큐 함수에 따라서 행동을 반환
    # 입실론 탐욕 정책에 따라서 행동을 반환
    def get_action(self, state):
        if np.random.rand() < self.epsilon:
            # 랜덤 행동
            action = np.random.choice(self.actions)
        else:
            # 큐 함수에 따른 행동
            next_state, q_value = self.possible_next_state(state)
            action = self.arg_max(q_value)
        # 행동 확률 값을 화면에 출력하기 위해 추가한 코드
        for i in range(0, 4):
            if i == int(action):
                percent = 1 - self.epsilon + self.epsilon / 4
                self.percentage_table[str(state)][i] = percent
            else:
                percent = self.epsilon / 4
                self.percentage_table[str(state)][i] = percent

        return int(action)

    # 후보가 여럿이면 arg_max를 계산하고 무작위로 하나를 반환
    @staticmethod
    def arg_max(next_state):
        max_index_list = []
        max_value = next_state[0]
        for index, value in enumerate(next_state):
            if value > max_value:
                max_index_list.clear()
                max_value = value
                max_index_list.append(index)
            elif value == max_value:
                max_index_list.append(index)
        return random.choice(max_index_list)

    # 목표 지점을 찾아 갈 수 있도록 보상을 체크
    def get_reward(self, col, row):
        if col == 4 and row  == 3:
            return 1  # 목표 지점의 보상
        elif [col, row] in [[4, 0], [2, 1], [4, 2], [3, 3], [5, 3], [2, 4], [3, 5]]:
            return -1  # 장해물의 보상
        else:
            return 0

    # 가능한 다음 모든 상태들을 반환
    def possible_next_state(self, state):
        col, row = state
        next_state_value = [0.0] * 4
        next_state_reward = [0] * 4  # 선택 가능한 행동에 대한 보상 배열
        # 상
        if row != 0:
            next_state_value[0] = self.value_table[str([col, row - 1])]
            next_state_reward[0] = self.get_reward(col, row - 1)
        else:
            next_state_value[0] = self.value_table[str(state)]
            next_state_reward[0] = self.get_reward(col, row)
        # 하
        if row != self.height - 1:
            next_state_value[1] = self.value_table[str([col, row + 1])]
            next_state_reward[1] = self.get_reward(col, row + 1)
        else:
            next_state_value[1] = self.value_table[str(state)]
            next_state_reward[1] = self.get_reward(col, row)
        # 좌
        if col != 0:
            next_state_value[2] = self.value_table[str([col - 1, row])]
            next_state_reward[2] = self.get_reward(col - 1, row)
        else:
            next_state_value[2] = self.value_table[str(state)]
            next_state_reward[2] = self.get_reward(col, row)
        # 우
        if col != self.width - 1:
            next_state_value[3] = self.value_table[str([col + 1, row])]
            next_state_reward[3] = self.get_reward(col + 1, row)
        else:
            next_state_value[3] = self.value_table[str(state)]
            next_state_reward[3] = self.get_reward(col, row)

        # 큐 함수 값 계산
        q_value = [0.0, 0.0, 0.0, 0.0]
        for i in range(4):
            q_value[i] = next_state_reward[i] + (self.discount_factor * next_state_value[i])

        return next_state_value, q_value


# 메인 함수
if __name__ == "__main__":
    episode_cnt = 0
    hazard_cnt = 0
    reward_cnt = 0

    env = Env()
    agent = MCAgent(actions=list(range(env.n_actions)))

    for episode in range(100):
        state = env.reset()
        action = agent.get_action(state)

        episode_cnt += 1
        while True:
            env.render()

            # 다음 상태로 이동
            # 보상은 숫자이고, 완료 여부는 boolean
            present_state, next_state, reward, done = env.step(action)
            agent.save_sample(present_state, reward, done)
            agent.save_reward(state, action, reward)


            # 다음 행동 받아옴
            next_action = agent.get_action(next_state)

            state = next_state
            action = next_action
            # 모든 큐함수를 화면에 표시
            env.print_value_q_all(agent.percentage_table, agent.value_table)

            # 에피소드가 완료됐을 때, 큐 함수 업데이트
            if done:
                if reward < 0 :
                    hazard_cnt += 1
                else:
                    reward_cnt += 1
                print(f'Episode = {episode_cnt}, Hazard = {hazard_cnt}, Reward = {reward_cnt}')
                agent.update()
                agent.samples.clear()
                break
