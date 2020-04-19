import gym
import torch
import random
import collections
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt


class ReplayBuffer():
    def __init__(self, buffer_maxlen):
        self.buffer = collections.deque(maxlen=buffer_maxlen)

    def push(self, data):
        self.buffer.append(data)

    def sample(self, batch_size):
        state_list = []
        action_list = []
        reward_list = []
        next_state_list = []
        done_list = []

        batch = random.sample(self.buffer, batch_size)
        for experience in batch:
            s, a, r, n_s, d = experience

            state_list.append(s)
            action_list.append(a)
            reward_list.append(r)
            next_state_list.append(n_s)
            done_list.append(d)

        return torch.FloatTensor(state_list).to(device), \
               torch.FloatTensor(action_list).to(device), \
               torch.FloatTensor(reward_list).unsqueeze(-1).to(device), \
               torch.FloatTensor(next_state_list).to(device), \
               torch.FloatTensor(done_list).unsqueeze(-1).to(device)

    def buffer_len(self):
        return len(self.buffer)


class critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(critic, self).__init__()
        self.linear1 = nn.Linear(state_dim + action_dim, 128)
        self.linear2 = nn.Linear(128, 128)
        self.linear3 = nn.Linear(128, 1)

    def forward(self, state, action):
        x = torch.cat((state, action), 1)
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = self.linear3(x)

        return x


class actor(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(actor, self).__init__()
        self.linear1 = nn.Linear(state_dim, 128)
        self.linear2 = nn.Linear(128, 128)
        self.linear3 = nn.Linear(128, action_dim)

    def forward(self, state):
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        x = torch.tanh(self.linear3(x))

        return x


class DDPG():
    def __init__(self, env, gamma, tau, buffer_maxlen, critic_lr, actor_lr):
        self.env = env
        self.state_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.shape[0]

        self.gamma = gamma
        self.tau = tau

        # buffer
        self.buffer = ReplayBuffer(buffer_maxlen)

        # network
        self.critic_net = critic(self.state_dim, self.action_dim).to(device)
        self.target_critic_net = critic(self.state_dim, self.action_dim).to(device)
        self.actor_net = actor(self.state_dim, self.action_dim).to(device)
        self.target_actor_net = actor(self.state_dim, self.action_dim).to(device)

        # optimizer
        self.critic_optimizer = optim.Adam(self.critic_net.parameters(), lr=critic_lr)
        self.actor_optimizer = optim.Adam(self.actor_net.parameters(), lr=actor_lr)

        # copy net parameters
        self.target_critic_net.load_state_dict(self.critic_net.state_dict())
        self.target_actor_net.load_state_dict(self.actor_net.state_dict())

    # action add  Gaussian noise
    def get_action(self, state, Episode):
        state = torch.FloatTensor(state).to(device)
        action = self.actor_net(state)*2
        action = action + np.clip(np.random.normal(0, 1), -0.25, 0.25)*np.exp(-Episode/10)
        action = torch.clamp(action, -2, 2)

        return action.detach().cpu().numpy()

    def update(self, batch_size):
        state, action, reward, next_state, done = self.buffer.sample(batch_size)
        next_action = self.target_actor_net(state).detach()

        # critic loss
        value = self.critic_net(state, action)
        target_value = reward + done * self.gamma * self.target_critic_net(next_state, next_action).detach()
        critic_loss = F.mse_loss(value, target_value)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # actor loss
        actor_loss = -self.critic_net(state, self.actor_net(state))

        self.actor_optimizer.zero_grad()
        actor_loss.mean().backward()
        self.actor_optimizer.step()

        # delay update target
        for target_param, param in zip(self.target_critic_net.parameters(), self.critic_net.parameters()):
            target_param.data.copy_(self.tau * param + (1 - self.tau) * target_param)

        for target_param, param in zip(self.target_actor_net.parameters(), self.actor_net.parameters()):
            target_param.data.copy_(self.tau * param + (1 - self.tau) * target_param)


def main(env, agent, Episode, batch_size):
    Return = []
    for episode in range(Episode):
        score = 0
        state = env.reset()
        for i in range(400):
            action = agent.get_action(state, Episode)
            next_state, reward, done, _ = env.step(action)
            done_mask = 0 if done else 1

            agent.buffer.push((state, action, reward, next_state, done_mask))
            state = next_state
            score += reward

            if done:
                break
            if agent.buffer.buffer_len() > 500:
                agent.update(batch_size)

        print("episode:{}, Return:{}, Buffer_len:{}".format(episode, score, agent.buffer.buffer_len()))
        Return.append(score)

    env.close()
    plt.plot(Return)
    plt.ylabel('Return')
    plt.xlabel("Episode")
    plt.grid(True)
    plt.show()


if __name__ == '__main__':
    env = gym.make("Pendulum-v0")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # param
    gamma = 0.98
    tau = 0.02
    buffer_maxlen = 50000
    critic_lr = 3e-3
    actor_lr = 3e-3

    batch_size = 128
    Episode = 60

    agent = DDPG(env, gamma, tau, buffer_maxlen, critic_lr, actor_lr)
    main(env, agent, Episode, batch_size)