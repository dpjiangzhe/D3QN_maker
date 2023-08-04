import torch as T
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as nnf
import numpy as np
import json


device = T.device("cpu")


def set_device(online=False):
    global device
    _device = T.device("cpu")
    if not online:
        if T.backends.mps.is_available():
            # device = T.device("mps" if T.backends.mps.is_available() else "cpu")
            _device = T.device("mps")
        elif T.cuda.is_available():
            # device = T.device("cuda:0" if T.cuda.is_available() else "cpu")
            _device = T.device("cuda:0")
    elif T.cuda.is_available():
        # device = T.device("cuda:0" if T.cuda.is_available() else "cpu")
        _device = T.device("cuda:0")
    device = _device
    return _device


class ReplayBuffer:
    def __init__(self, state_dim, action_dim, max_size, batch_size):
        self.mem_size = max_size
        self.batch_size = batch_size
        self.mem_cnt = 0

        self.state_memory = np.zeros((self.mem_size, state_dim))
        self.action_memory = np.zeros((self.mem_size,))
        self.reward_memory = np.zeros((self.mem_size,))
        self.next_state_memory = np.zeros((self.mem_size, state_dim))
        self.terminal_memory = np.zeros((self.mem_size,), dtype=np.bool_)

    def store_transition(self, state, action, reward, state_, done):
        mem_idx = self.mem_cnt % self.mem_size

        self.state_memory[mem_idx] = state
        self.action_memory[mem_idx] = action
        self.reward_memory[mem_idx] = reward
        self.next_state_memory[mem_idx] = state_
        self.terminal_memory[mem_idx] = done

        self.mem_cnt += 1
        # print("store:", state, action, reward, state_)

    def sample_buffer(self):
        mem_len = min(self.mem_size, self.mem_cnt)

        batch = np.random.choice(mem_len, self.batch_size, replace=False)

        states = self.state_memory[batch]
        actions = self.action_memory[batch]
        rewards = self.reward_memory[batch]
        states_ = self.next_state_memory[batch]
        terminals = self.terminal_memory[batch]

        return states, actions, rewards, states_, terminals

    def ready(self):
        return self.mem_cnt > self.batch_size


class DuelingDeepQNetwork(nn.Module):
    def __init__(self, alpha, state_dim, action_dim, fc_dims, dropout=0.0):
        super(DuelingDeepQNetwork, self).__init__()

        self.fc = []
        self.fc_dims = fc_dims
        _dim0 = state_dim
        for i, _dim in enumerate(fc_dims):
            _fc = nn.Linear(_dim0, _dim).to(device)
            self.fc.append(_fc)
            _dim0 = _dim
            key = "fc" + str(i)
            setattr(self, key, _fc)

        # dropout
        if dropout > 0:
            self.dropout = nn.Linear(p=dropout)
        else:
            self.dropout = None

        self.V = nn.Linear(_dim0, 1).to(device)
        self.A = nn.Linear(_dim0, action_dim).to(device)

        self.optimizer = optim.Adam(self.parameters(), lr=alpha)
        self.to(device)

    def forward(self, state):
        x = state
        for _fc in self.fc:
            x = T.relu(_fc(x))
            # dropout
            if self.dropout is not None:
                x = self.dropout(x)

        V = self.V(x).to(device)
        A = self.A(x).to(device)
        Q = V + A - T.mean(A, dim=-1, keepdim=True)

        # print(self.dropout, len(state), Q)

        return Q

    def save_checkpoint(self, checkpoint_file):
        T.save(self.state_dict(), checkpoint_file)
        print("saving model:", self.fc[0].state_dict(), self.fc0.state_dict())

    def load_checkpoint(self, checkpoint_file):
        print("init model:", self.fc[0].state_dict(), self.fc0.state_dict())
        self.load_state_dict(T.load(checkpoint_file))
        self.eval()
        print("loading model:", self.fc[0].state_dict(), self.fc0.state_dict())


class D3QN:
    def __init__(self, config, model_path):
        self.name = "D3QN"
        self.alpha = 0.0003
        self.gamma = 0.99
        self.tau = 0.005
        self.epsilon = 1.0
        self.eps_min = 0.01
        self.eps_dec = 5e-7
        self.batch_size = 256
        self.ckpt_dir = model_path
        self.action_space = []
        self.action_dim = 2
        self.action_edge = 4
        self.state_dim = 256
        self.need_coding = True
        self.memory_capacity = 1000000
        self.eval_dims = []
        self.target_dims = []
        self.dropout = 0.0

        for key, value in config.items():
            assert getattr(self, key) is not None
            setattr(self, key, value)

        self.action_space = [i for i in range(self.action_dim)]

        if self.eval_dims is None or len(self.eval_dims) == 0:
            self.eval_dims = [256, 256]
        self.q_eval = DuelingDeepQNetwork(alpha=self.alpha, state_dim=self.state_dim, action_dim=self.action_dim,
                                          fc_dims=self.eval_dims, dropout=self.dropout)
        if self.target_dims is None or len(self.target_dims) == 0:
            self.target_dims = [256, 256]
        self.q_target = DuelingDeepQNetwork(alpha=self.alpha, state_dim=self.state_dim, action_dim=self.action_dim,
                                            fc_dims=self.target_dims, dropout=self.dropout)

        self.pointer = 0
        self.memory = ReplayBuffer(state_dim=self.state_dim, action_dim=self.action_dim,
                                   max_size=self.memory_capacity, batch_size=self.batch_size)

        self.update_network_parameters(tau=1.0)

        self.q_eval.to(device)
        self.q_target.to(device)

    def update_network_parameters(self, tau=None):
        if tau is None:
            tau = self.tau

        for q_target_params, q_eval_params in zip(self.q_target.parameters(), self.q_eval.parameters()):
            q_target_params.data.copy_(tau * q_eval_params + (1 - tau) * q_target_params)

    def remember(self, state, action, reward, state_, done):
        self.memory.store_transition(state, action, reward, state_, done)
        self.pointer += 1

    def decrement_epsilon(self):
        self.epsilon = self.epsilon - self.eps_dec \
            if self.epsilon > self.eps_min else self.eps_min

    def choose_action(self, observation, is_training=True):
        state = T.tensor(np.array(observation), dtype=T.float).to(device)
        q_vals = self.q_eval.forward(state)
        action = T.argmax(q_vals).item()

        if (np.random.random() < self.epsilon) and is_training:
            action = np.random.choice(self.action_space)

        return action

    def learn(self):
        if not self.memory.ready():
            return

        states, actions, rewards, next_states, terminals = self.memory.sample_buffer()
        batch_idx = T.arange(self.batch_size, dtype=T.long).to(device)
        states_tensor = T.tensor(states, dtype=T.float).to(device)
        actions_tensor = T.tensor(actions, dtype=T.long).to(device)
        rewards_tensor = T.tensor(rewards, dtype=T.float).to(device)
        next_states_tensor = T.tensor(next_states, dtype=T.float).to(device)
        terminals_tensor = T.tensor(terminals).to(device)

        # print(len(states), len(actions), len(rewards), len(next_states))
        # print(actions)

        with T.no_grad():
            q_ = self.q_target.forward(next_states_tensor)
            max_actions = T.argmax(self.q_eval.forward(next_states_tensor), dim=-1)
            q_[terminals_tensor] = 0.0
            target = rewards_tensor + self.gamma * q_[batch_idx, max_actions]
        q = self.q_eval.forward(states_tensor)[batch_idx, actions_tensor]
        # print(batch_idx, actions_tensor, q)

        loss = nnf.mse_loss(q, target.detach())
        loss_value = loss
        self.q_eval.optimizer.zero_grad()
        loss.backward()
        self.q_eval.optimizer.step()

        self.update_network_parameters()
        self.decrement_epsilon()
        return loss_value

    def save_models(self, filepath="", episode=0, keys={}):
        if filepath == "":
            filepath = self.ckpt_dir
        self.q_eval.save_checkpoint(filepath + 'D3QN_q_eval_{}.pth'.format(episode))
        print('Saving Q_eval network successfully!')
        self.q_target.save_checkpoint(filepath + 'D3QN_Q_target_{}.pth'.format(episode))
        print('Saving Q_target network successfully!')
        if len(keys) > 0:
            json_str = json.dumps(keys)
            with open(filepath + "feature_keys.json", 'w') as keyfile:
                keyfile.write(json_str)

    def load_models(self, filepath="", episode=0):
        if filepath == "":
            filepath = self.ckpt_dir
        self.q_eval.load_checkpoint(filepath + 'D3QN_q_eval_{}.pth'.format(episode))
        print('Loading Q_eval network successfully!')
        self.q_target.load_checkpoint(filepath + 'D3QN_Q_target_{}.pth'.format(episode))
        print('Loading Q_target network successfully!')

        self.q_eval.to(device)
        self.q_target.to(device)

