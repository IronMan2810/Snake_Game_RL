import torch
import torch.nn as nn
import torch.optim as optim


class QTrainer:
    def __init__(self, model, lr, gamma): #initializing 
        self.lr = lr
        self.gamma = gamma
        self.model = model
        self.optimizer = optim.Adam(model.parameters(), lr=self.lr) #optimizer
        self.criterion = nn.MSELoss() #loss function

    def train_step(self, state, action, reward, next_state, done): #trainer
        state = torch.tensor(state, dtype=torch.float)
        next_state = torch.tensor(next_state, dtype=torch.float)
        action = torch.tensor(action, dtype=torch.long)
        reward = torch.tensor(reward, dtype=torch.float)

        if len(state.shape) == 1: #if there 1 dimension
            state = torch.unsqueeze(state, 0)
            next_state = torch.unsqueeze(next_state, 0)
            action = torch.unsqueeze(action, 0)
            reward = torch.unsqueeze(reward, 0)
            done = (done,)
        pred = self.model(state) #using the Q=model predict equation above

        target = pred.clone() #using Qnew = r+y(next predicted Q) as mentionned above
        for idx in range(len(done)):
            Q_new = reward[idx]
            if not done[idx]:
                Q_new = reward[idx] + self.gamma * torch.max(self.model(next_state[idx]))
            target[idx][torch.argmax(action[idx]).item()] = Q_new

        self.optimizer.zero_grad() #calculating loss function
        loss = self.criterion(target, pred)
        loss.backward()
        self.optimizer.step()