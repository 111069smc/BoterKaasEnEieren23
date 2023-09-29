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

#alpha is de leerfactor van de agent. Deze bepaalt hoe snel de agent nieuwe kennis adopteert. Hoe hoger dit getal, hoe sneller de agent geneigd zal zijn om oude kennis te vervangen door nieuwe kennis.
#epsilon is de verkenningsfactor van de agent. Deze bepaalt hoe vaak de agent nieuwe dingen probeert. Hoe hoger dit getal, hoe vaker de agent een willekeurige actie probeert in plaats van de best bekende zet.
my_agent = MyAgent(alpha=0.1, epsilon=0.5)
random_agent = RandomAgent()

train_and_plot(
    agent=my_agent,
    validation_agent=random_agent,
    iterations=100,
    trainings=100,
    validations=1000)

#agent wordt opnieuw aageroepen
my_agent = load('MyAgent_3000')
my_agent.learning = False

validation_agent = RandomAgent()

validation_result = validate(agent_x=my_agent, agent_o=validation_agent, iterations=100)
 
plot_validation(validation_result)

#train functie aanroepen
#train(my_agent, 3000)

save(my_agent, 'MyAgent_3000')

start(player_x=my_agent)


