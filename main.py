import random
 
from bke import MLAgent, is_winner, opponent, RandomAgent, train_and_plot, load, save
 
 
class MyAgent(MLAgent):
    def evaluate(self, board):
        if is_winner(board, self.symbol):
            reward = 1
        elif is_winner(board, opponent[self.symbol]):
            reward = -1
        else:
            reward = 0
        return reward



for a in range(1, 5):
    for e in range(1, 11):
        random.seed(1)
        print("a is ", (a/100))
        print("e is ", (e/100))
        my_agent = MyAgent(alpha=(1/10), epsilon=(10/10))
        #my_agent = load('MyAgent_3000')
        #my_agent.learning = False

        random_agent = RandomAgent()
        train_and_plot(
            agent=my_agent,
            validation_agent=random_agent,
            iterations=85,
            trainings=100,
            validations=2500)

random.seed(1)

my_agent = MyAgent(alpha=(0.03), epsilon=(0.5))
#my_agent = load('MyAgent_3000')
#my_agent.learning = False

random_agent = RandomAgent()
train_and_plot(
    agent=my_agent,
    validation_agent=random_agent,
    iterations=50,
    trainings=100,
    validations=1000)