import random

from bke import MLAgent, is_winner, opponent, load, validate, RandomAgent, plot_validation, train, train_and_plot, save, start


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

my_agent = MyAgent(alpha=0.8, epsilon=0.2)
random_agent = RandomAgent()

train_and_plot(
    agent=my_agent,
    validation_agent=random_agent,
    iterations=20,
    trainings=100,
    validations=1000)

#agent worst opnieuw aageroepen
my_agent = load('MyAgent_3000')
my_agent.learning = False

validation_agent = RandomAgent()

validation_result = validate(agent_x=my_agent, agent_o=validation_agent, iterations=100)
 
plot_validation(validation_result)

#train functie aanroepen
train(my_agent, 3000)

save(my_agent, 'MyAgent_3000')

start(player_x=my_agent)


