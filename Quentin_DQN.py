import random
import numpy as np
import torch
from collections import deque

# if GPU is to be used
device = torch.device(
    "cuda" if torch.cuda.is_available() else
    "mps" if torch.backends.mps.is_available() else
    "cpu"
)

class DQN(torch.nn.Module):
    def __init__(self, state_size, action_size, kernel_dim=64):
        super(DQN, self).__init__()
        self.layer1 = torch.nn.Linear(state_size, kernel_dim)
        self.layer2 = torch.nn.Linear(kernel_dim, kernel_dim)
        self.layer3 = torch.nn.Linear(kernel_dim, action_size)
        # Apply He initialization to the layers
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, torch.nn.Linear):
                torch.nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    torch.nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = torch.nn.functional.relu(self.layer1(x))
        x = torch.nn.functional.relu(self.layer2(x))
        return self.layer3(x)


class QuentinDQNAgent:
    def __init__(self, size, state_size, action_size, **kwargs):
        self.size = size
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=10000)
        self.gamma = kwargs.get('gamma', 0.95)
        self.epsilon = kwargs.get('epsilon', 1.0)
        self.epsilon_max = kwargs.get('epsilon_max', 1.0)
        self.epsilon_min = kwargs.get('epsilon_min', 0.01)
        self.epsilon_decay = kwargs.get('epsilon_decay', 0.995)
        self.learning_rate = kwargs.get('learning_rate', 0.02)
        self.tau = kwargs.get('tau', 0.005)
        self.model = DQN(state_size, action_size, kernel_dim=kwargs.get('kernel_size', 64)).to(device)
        self.target_model = DQN(state_size, action_size, kernel_dim=kwargs.get('kernel_size', 64)).to(device)
        self.update_target_model()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        # self.criterion = torch.nn.MSELoss()
        self.criterion = torch.nn.SmoothL1Loss()

    def update_target_model(self):
        # self.target_model.load_state_dict(self.model.state_dict())
        target_model_state_dict = self.target_model.state_dict()
        model_state_dict = self.model.state_dict()
        for key in model_state_dict:
            target_model_state_dict[key] = model_state_dict[key]*self.tau + target_model_state_dict[key]*(1-self.tau)
        self.target_model.load_state_dict(target_model_state_dict)

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state, unavail):
        available_actions = list(set(range(self.action_size)) - set(unavail))
        if not available_actions:
            return None
        if np.random.rand() <= self.epsilon:
            return random.choice(available_actions)
        act_values = self.model(torch.FloatTensor(state))
        available_q_values = act_values[available_actions]
        return available_actions[torch.argmax(available_q_values).item()]

    def epsilon_update(self, episode):
      return self.epsilon_min + (self.epsilon_max - self.epsilon_min) * np.exp(-self.epsilon_decay * episode)
    
    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target += self.gamma * torch.max(self.target_model(torch.FloatTensor(next_state))).item()
            target_f = self.model(torch.FloatTensor(state))
            target_f[action] = target
            self.optimizer.zero_grad()
            loss = self.criterion(target_f, self.model(torch.FloatTensor(state)))
            loss.backward()
            self.optimizer.step()
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def save(self, name):
        torch.save(self.model.state_dict(), name)


def load_model(model_path, state_size, action_size, kernel_size=128):
    model = DQN(state_size, action_size, kernel_size)
    model.load_state_dict(torch.load(model_path))
    model.eval()  # Set the model to evaluation mode
    return model

def convert_state(board, size, player):
    state = np.zeros((size * size + 1))
    for i, val in enumerate(board):
        state[i] = val
    state[-1] = player
    return state

def rank_actions(model, state):
    state_tensor = torch.FloatTensor(state).unsqueeze(0)  # Add batch dimension
    with torch.no_grad():
        q_values = model(state_tensor).numpy().flatten()
    ranked_actions = np.argsort(q_values)[::-1]  # Sort actions by Q-value in descending order
    q_values = q_values[ranked_actions]
    return ranked_actions, q_values