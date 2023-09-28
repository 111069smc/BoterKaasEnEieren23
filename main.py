import random
from bke import MLAgent, is_winner, opponent, RandomAgent, train_and_plot

class MyAgent(MLAgent):
    def evaluate(self, board):
        if is_winner(board, self.symbol):
            reward = 1
        elif is_winner(board, opponent[self.symbol]):
            reward = -1
        else:
            reward = 0
        return reward

random.seed(1)

my_agent = MyAgent() 
random_agent = RandomAgent()

#hier wordt de functie aangeroepen die traint en tegelijk de voortgang plot
train_and_plot(
    agent=my_agent,
    validation_agent=random_agent,
    iterations=50,
    trainings=100,
    validations=1000)

#iteration betekent herhaling
#Het argument trainings geeft per herhaling aan hoeveel er getraind wordt.
#validations komt neer op het aantal validatiespellen dat per herhaling gespeeld wordt