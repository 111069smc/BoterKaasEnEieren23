from bke import MLAgent, is_winner, opponent, load, validate, RandomAgent, plot_validation

class MyAgent(MLAgent):
    def evaluate(self, board):
        if is_winner(board, self.symbol):
            reward = 1
        elif is_winner(board, opponent[self.symbol]):
            reward = -1
        else:
            reward = 0
        return reward

# de agent wordt opnieuw ingeladen met de functie load
my_agent = load('MyAgent_3000') 

my_agent.learning = False

validation_agent = RandomAgent()

validation_result = validate(agent_x=my_agent, agent_o=validation_agent, iterations=100)

#een functie wordt aangeroepen om de validatieresultaten te plotten
plot_validation(validation_result)