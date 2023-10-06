import torch
import torch.nn as nn
import torch.optim as optim
import os

class DoubleLinear_QNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = torch.relu(self.linear1(x))
        x = self.linear2(x)
        return x

    def save(self, file_name='model.pth'):
        model_folder_path = './model'
        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)

        file_name = os.path.join(model_folder_path, file_name)
        torch.save(self.state_dict(), file_name)


class DoubleQTrainer:
    def __init__(self, model, target_model, lr, gamma):
        self.lr = lr
        self.gamma = gamma
        self.model = model
        self.target_model = target_model
        self.optimizer = optim.Adam(model.parameters(), lr=self.lr)
        self.criteria = nn.MSELoss()

    def train_step(self, state, action, reward, next_state, done):
        state = torch.tensor(state, dtype=torch.float)
        next_state = torch.tensor(next_state, dtype=torch.float)
        action = torch.tensor(action, dtype=torch.long)

        # 1: Predicted Q values with the current state using the model
        pred = self.model(state)

        # 2: Predict Q values for the next state using the target model
        target_next_state_values = self.target_model(next_state).detach()

        # 3: Compute the target Q values using Double DQN formula
        target = pred.clone()
        for i in range(len(done)):
            Q_new = reward[i]
            if not done[i]:
                best_action = torch.argmax(self.model(next_state[i]))
                Q_new = reward[i] + self.gamma * target_next_state_values[i][best_action]

            target[i][action[i]] = Q_new

        # 4: Compute the mean squared error loss
        loss = self.criteria(target, pred)

        # 5: Perform backpropagation and optimization
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()