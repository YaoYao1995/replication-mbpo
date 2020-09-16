import gym
import torch
import numpy as np
class EnvSampler():
    def __init__(self, env, max_path_length=1000):
        self.env = env
        self.device = torch.device("cuda")
        self.path_length = 0
        self.current_state = None
        self.max_path_length = max_path_length
        self.path_rewards = []
        self.sum_reward = 0
        self.agents_num = 1

    def sample(self, agent, eval_t=False):
        if self.current_state is None:
            self.current_state = self.env.reset()

        cur_state = self.current_state
        action = agent.select_action(self.current_state, eval_t)
        next_state, reward, terminal, info = self.env.step(action)
        self.path_length += 1
        self.sum_reward += reward

        # TODO: Save the path to the env_pool
        if terminal or self.path_length >= self.max_path_length:
            self.current_state = None
            self.path_length = 0
            self.path_rewards.append(self.sum_reward)
            self.sum_reward = 0
            
        else:
            self.current_state = next_state

        return cur_state, action, next_state, reward, terminal, info
    
    def ensemble_mean_sample(self, agents, eval_t=True):
        if self.current_state is None:
            self.current_state = self.env.reset()

        cur_state = self.current_state
        actions = []
        for agent in agents:
            action_ = agent.select_action(cur_state, eval_t)
            actions.append(action_)
        action = np.mean(actions,axis=0)
        next_state, reward, terminal, info = self.env.step(action)
        self.path_length += 1
        self.sum_reward += reward

        # TODO: Save the path to the env_pool
        if terminal or self.path_length >= self.max_path_length:
            self.current_state = None
            self.path_length = 0
            self.path_rewards.append(self.sum_reward)
            self.sum_reward = 0
        else:
            self.current_state = next_state

        return cur_state, action, next_state, reward, terminal, info
    
    def ensemble_ucb_sample(self, agents, eval_t=False):
        if self.current_state is None:
            self.current_state = self.env.reset()

        cur_state = self.current_state
        # select action from candidates by ucb
        actions = []
        for agent in agents:
            action_ = agent.select_action(cur_state, eval_t)
            actions.append(action_)
        actions = np.array(actions)
        self.agents_num = len(agents)
        
        states = np.array([cur_state])
        states = states.repeat(self.agents_num, axis=0))
        state_batch = torch.FloatTensor(states).to(self.device)
        action_batch = torch.FloatTensor(actions).to(self.device)

        ensemble_q_value = np.zeros((self.agents_num, self.agents_num, 2))
        for index, agent in enumerate(agents):
            qf1, qf2 = agent.critic(state_batch, action_batch)
            ensemble_q_value[index] = torch.cat((qf1, qf2), axis=-1).detach().cpu().numpy()
        q_value_estimate_ = np.mean(ensemble_q_value, -1)
        q_value_uncertainty = np.std(q_value_estimate_, 0)
        q_value_estimate = np.mean(q_value_estimate_, 0)
        #action = agent.select_action(self.current_state, eval_t)

        next_state, reward, terminal, info = self.env.step(action)
        self.path_length += 1
        self.sum_reward += reward

        # TODO: Save the path to the env_pool
        if terminal or self.path_length >= self.max_path_length:
            self.current_state = None
            self.path_length = 0
            self.path_rewards.append(self.sum_reward)
            self.sum_reward = 0
        else:
            self.current_state = next_state

        return cur_state, action, next_state, reward, terminal, info
