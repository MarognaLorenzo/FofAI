# multiAgents.py
# --------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent
from pacman import GameState

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """

    def getAction(self, gameState: GameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState: GameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newGhostPositions = [state.getPosition() for state in newGhostStates]
        
        starting_score = 0

        def closestElementDistance(pacman_pos, elements):
            print(len(elements))
            def dist(a, b):
                return abs(a[0] - b[0]) + abs(a[1] - b[1])
            distances = [dist(pacman_pos, elem) for elem in elements]
            res = min(distances)
            print("min dist: ", res)
            return res

        def expDecaying(dist, relevance = 10):
            return relevance * pow(relevance, -dist)

        closestGhostDist = closestElementDistance(newPos, newGhostPositions)

        starting_score -= expDecaying(closestGhostDist)

        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        if newPos in newFood.asList():
            starting_score += 20


        closestFood = closestElementDistance(newPos, currentGameState.getFood().asList())
        print("closestFood: " , closestFood)
        starting_score += expDecaying(closestFood)


        print("Score: ", starting_score)

        return starting_score

def scoreEvaluationFunction(currentGameState: GameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """ 

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        def minimax_value(gameState: GameState, agentIndex: int, depth: int):
            if gameState.isWin() or gameState.isLose():
                depth = 0

            if depth == 0:
                return self.evaluationFunction(gameState)

            if agentIndex == 0:
                actions = gameState.getLegalActions(0)
                return max([minimax_value(gameState.generateSuccessor(agentIndex, action), 1, depth) for action in actions])
            else:
                nextAgentIndex = (agentIndex + 1) % gameState.getNumAgents()
                newDepth = depth if nextAgentIndex != 0 else depth - 1

                actions = gameState.getLegalActions(agentIndex)
                return min([minimax_value(gameState.generateSuccessor(agentIndex, action), nextAgentIndex, newDepth) for action in actions])


        possible_actions = gameState.getLegalActions(0)
        scenarios = [(minimax_value(gameState.generateSuccessor(0, action), agentIndex=1, depth=self.depth), action) for action in possible_actions]
        return max(scenarios)[1]
    

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        
        def minimax_value(gameState: GameState, agentIndex: int, depth: int, alpha: float, beta: float):
            if depth == 0 or gameState.isWin() or gameState.isLose():
                return self.evaluationFunction(gameState)

            nextGameState = lambda action : gameState.generateSuccessor(agentIndex, action)
            actions = gameState.getLegalActions(agentIndex)
            nextAgentIndex = (agentIndex + 1) % gameState.getNumAgents()

            if agentIndex == 0:
                bestValue = float('-inf')
                for action in actions:
                    bestValue = max(bestValue, minimax_value(nextGameState(action), nextAgentIndex, depth, alpha=alpha, beta=beta))
                    if bestValue > beta:
                        break
                    alpha = max(alpha, bestValue)
                return bestValue

            else:
                newDepth = depth if nextAgentIndex != 0 else depth - 1
                bestValue = float('inf')
                for action in actions:
                    bestValue = min(bestValue, minimax_value(nextGameState(action), nextAgentIndex, newDepth, alpha=alpha, beta=beta))
                    if bestValue < alpha:
                        break
                    beta = min(beta, bestValue)
                return bestValue

        alpha = float('-inf')
        beta = float('inf')

        bestValue = float('-inf')
        actions = gameState.getLegalActions(0)
        best_action = None
        for action in actions:
            if (value :=  minimax_value(gameState.generateSuccessor(0, action), 1, self.depth, alpha, beta)) > bestValue:
                alpha = max(alpha, value)
                bestValue = value
                best_action = action
        return best_action

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        
        def minimax_value(gameState: GameState, agentIndex: int, depth: int):
            if depth == 0 or gameState.isWin() or gameState.isLose():
                return self.evaluationFunction(gameState)

            nextGameState = lambda action : gameState.generateSuccessor(agentIndex, action)
            actions = gameState.getLegalActions(agentIndex)
            nextAgentIndex = (agentIndex + 1) % gameState.getNumAgents()

            if agentIndex == 0:
                    return max(minimax_value(nextGameState(action), nextAgentIndex, depth) for action in actions)

            else:
                newDepth = depth if nextAgentIndex != 0 else depth - 1
                return sum(minimax_value(nextGameState(action), nextAgentIndex, newDepth) for action in actions) / len(actions)

        bestValue = float('-inf')
        actions = gameState.getLegalActions(0)
        best_action = None
        for action in actions:
            if (value :=  minimax_value(gameState.generateSuccessor(0, action), 1, self.depth)) > bestValue:
                bestValue = value
                best_action = action
        return best_action

def betterEvaluationFunction(currentGameState: GameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    print()
    def dist(a, b):
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    score = currentGameState.getScore()

    if currentGameState.isWin() or currentGameState.isLose():
        return currentGameState.getScore()
    score -= sum(dist(currentGameState.getPacmanPosition(), food) for food in currentGameState.getFood().asList())

    def closestElementDistance(pacman_pos, elements):
        distances = [dist(pacman_pos, elem) for elem in elements]
        res = min(distances)
        print("min dist: ", res, "elements: ", len(elements))
        return res

    def expDecaying(dist, relevance = 3):
        return relevance * pow(relevance, -dist+1)

    closestGhostDist = closestElementDistance(currentGameState.getPacmanPosition(), currentGameState.getGhostPositions())
    decrementing_ghost = expDecaying(closestGhostDist)
    print("decrementing_ghost: ", decrementing_ghost)
    score -= decrementing_ghost

    closestFood = closestElementDistance(currentGameState.getPacmanPosition(), currentGameState.getFood().asList())
    incrementing_food = expDecaying(closestFood)
    print("incrementing_food: ", incrementing_food)
    score += incrementing_food
    # print("closestFood: " , closestFood)


    print("Score: ", score)
    score += score * (random.randint(-5,5) / 10000)
    print("Modified score: ", score)
    
    return score


# Abbreviation
better = betterEvaluationFunction
