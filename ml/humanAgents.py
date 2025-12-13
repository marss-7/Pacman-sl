from game import Agent
from keyboardAgents import KeyboardAgent

class humanAgent(Agent):
    def __init__(self, **args):
        self.keyboard_agent = KeyboardAgent(**args)

    def getAction(self, state):
        return self.keyboard_agent.getAction(state)
