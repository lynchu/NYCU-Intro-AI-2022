from util import manhattanDistance
from game import Directions
import random, util
from game import Agent
import math

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """

    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.
        getAction chooses among the best options according to the evaluation function.
        
        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and child states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        The evaluation function takes in the current and proposed child
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.
        """
        # Useful information you can extract from a GameState (pacman.py)
        childGameState = currentGameState.getPacmanNextState(action)
        newPos = childGameState.getPacmanPosition()
        newFood = childGameState.getFood()
        newGhostStates = childGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        minGhostDistance = min([manhattanDistance(newPos, state.getPosition()) for state in newGhostStates])

        scoreDiff = childGameState.getScore() - currentGameState.getScore()

        pos = currentGameState.getPacmanPosition()
        nearestFoodDistance = min([manhattanDistance(pos, food) for food in currentGameState.getFood().asList()])
        newFoodsDistances = [manhattanDistance(newPos, food) for food in newFood.asList()]
        newNearestFoodDistance = 0 if not newFoodsDistances else min(newFoodsDistances)
        isFoodNearer = nearestFoodDistance - newNearestFoodDistance

        direction = currentGameState.getPacmanState().getDirection()
        if minGhostDistance <= 1 or action == Directions.STOP:
            return 0
        if scoreDiff > 0:
            return 8
        elif isFoodNearer > 0:
            return 4
        elif action == direction:
            return 2
        else:
            return 1


def scoreEvaluationFunction(currentGameState):
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
    Your minimax agent (Part 1)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        gameState.getNextState(agentIndex, action):
        Returns the child game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
    # Begin your code (Part 1)
        """
        * for part(a), 
        Take PACMAN as the agent for the most upper layer, and get possible actions.
        With currentState and all the possible actions for PACMAN, all the reachable nextStates are obtained
        Evalute the score of all nextStates, by calling ghost 1.
        Take the index of maxScore, return the action with maxScore.

        * for part(b),
        When reach terminal states(depth=0, win, lose), return the score of the state.
        Classify cases to 2 cases, PACMAN turn and GHOST turn, and assign value to parameters.
        Get possible actions of currentState.
        Obtain all the reachable nextStates, with agentIndex, currentState and all the possible actions.
        Evaluate the score of nextStates, by recursive the next state with minimaxAgent.
        Return the related bestScore, Max for pacman, min for all ghosts.
        Recursve down the depth until return bestScore.
        """
        # part(a)
        legalActions = gameState.getLegalActions(0) 
        successors = (gameState.getNextState(0, action) for action in legalActions)
        scores = [self._minimax(successor, 1, self.depth) for successor in successors]
        i = scores.index((max(scores)))
        return legalActions[i]
    
    # part(b)
    def _minimax(self, state, agentIndex: int, depth: int):
        if depth == 0 or state.isWin() or state.isLose():
            return self.evaluationFunction(state)
        
        if agentIndex == 0:  # PACMAN TURN
            selectBestScore = max 
            nextAgent = 1
            nextDepth = depth
        else:  # GHOST TURN
            selectBestScore = min
            nextAgent = (agentIndex + 1) % state.getNumAgents()
            nextDepth = (depth - 1) if nextAgent == 0 else depth

        legalActions = state.getLegalActions(agentIndex) 
        successors = (state.getNextState(agentIndex, action) for action in legalActions) 
        scores = [self._minimax(successor, nextAgent, nextDepth) for successor in successors]
        return selectBestScore(scores)
    # End your code (Part 1)


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (Part 2)
    Returns the minimax action using self.depth and self.evaluationFunction
    """
    def getAction(self, gameState):
    # Begin your code (Part 2)
        """
        * for part(a),
        Call the function of AlphaBetaAgent taking PACMAN as agent, and Return the best action.

        alphabeta is a recursive function implementint Alpha-Beta Pruning.
        If terminal state or max depth reached, return tuple(evaluatedScore, action)

        * for part(b), which is PACMAN turn, maximize the value
        First, Initialize the variable and Get possible actions of currentState,
        Then, go through all possible actions doing
        - Get the nextState with related action
        - Get the bestScore of nextState with recursion
        - Renew the maxScore 
        - Do the Pruning if maxScore > beta
        - Renew alpha and maxAction if maxScore>alpha

        * for part(c), which is GHOST turn, minimize the value for all ghosts
        First, Initialize the variable and Get possible actions of currentState,
        Then, go through all possible actions doing
        - Get the nextState with related action
        - Get the bestScore of nextState with recursion
        - Renew the minScore 
        - Do the Pruning if minScore < alpha
        - Renew beta and minAction if minScore>beta
        """
        # part(a)
        return self.alphabeta(gameState, 0, self.depth, -math.inf, math.inf)[1]

    def alphabeta(self, gameState, agentIndex, depth, alpha, beta):
        if depth == 0 or gameState.isWin() or gameState.isLose():
            return tuple([self.evaluationFunction(gameState), None])

        # part(b): PACMAN turn, maximize the value
        if agentIndex == 0:
            maxScore = -math.inf
            maxAction = None
            legalActions = gameState.getLegalActions(agentIndex)

            for action in legalActions:
                successor = gameState.getNextState(agentIndex, action)
                score = self.alphabeta(successor, 1, depth, alpha, beta)[0]
                maxScore = max(maxScore, score)
                if maxScore > beta:
                    return tuple([maxScore, action])
                maxAction = action if alpha<maxScore else maxAction
                alpha = max(alpha, maxScore)
            return tuple([maxScore, maxAction])
        
        # part(c): GHOST turn, minimize the value for all ghosts
        else:
            minScore = math.inf
            minAction = None
            legalActions = gameState.getLegalActions(agentIndex)
            nextAgent = (agentIndex + 1) % gameState.getNumAgents()
            nextDepth = depth if nextAgent > 0 else depth - 1
            for action in legalActions:
                successor = gameState.getNextState(agentIndex, action)
                score = self.alphabeta(successor, nextAgent, nextDepth, alpha, beta)[0]
                minScore = min(minScore, score)
                if minScore < alpha:
                    return tuple([minScore, action])
                minAction = action if beta>minScore else minAction
                beta = min(beta, minScore)
            return tuple([minScore, minAction])
    # End your code (Part 2)


class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (Part 3)
    """
    def getAction(self, gameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction
        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
    # Begin your code (Part 3)
        """
        * for part(a), 
        Take PACMAN as the agent for the most upper layer, and get possible actions.
        Obtain all the reachable nextStates with currentState and all the possible actions for PACMAN.
        Evalute the score of all nextStates by calling ghost 1.
        Take the index of maxScore, return the action with maxScore.

        * for part(b),
        When reach terminal states(depth=0, win, lose), return the score of the state.
        Get possible actions of currentState.
        Obtain all the reachable nextStates, with agentIndex, currentState and all the possible actions.
        Assign related value to nextAgent and nextDepth.
        Evaluate the score of nextStates, by recursive the next state with ExpectimaxAgent.
        Return the related bestScore, Max for pacman, AverageScore for all ghosts.
        Recursve down the depth until return bestScore.
        """
        # part(a)
        legalActions = gameState.getLegalActions(0)
        successors = (gameState.getNextState(0, action) for action in legalActions)
        scores = [self._expected(successor, 1, self.depth) for successor in successors]
        i = scores.index((max(scores)))
        return legalActions[i]

    # part(b)
    def _expected(self, state, agentIndex: int, depth: int):
        if depth == 0 or state.isWin() or state.isLose():
            return self.evaluationFunction(state)

        legalActions = state.getLegalActions(agentIndex)
        successors = (state.getNextState(agentIndex, action) for action in legalActions)
        nextAgent = (agentIndex + 1) % state.getNumAgents()
        nextDepth = depth if nextAgent > 0 else depth - 1
        scores = [self._expected(successor, nextAgent, nextDepth) for successor in successors]
        if agentIndex == 0:  # pacman turn
            return max(scores)
        else:  # ghost turn
            return sum(scores) / len(legalActions)
    # End your code (Part 3)


def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable evaluation function (Part 4).
    """
    """
    betterEvaluationFunction takes ghost-hunting and food-gobbling into account.
    Get originalScore with original EvaluationFunction.
    * for ghostPenalty:
    When Pacman eats the big dot, Pacman has to be awared of the timer counted to 0,
    since ghosts are able to eat pacman if the two encounterd.
    Get distanceFromUnscared with manhattan distance between ghost and pacman, and scaredTimer.
    Adapt the score by minusing ghostPenalty, which is the sum of (300 / square of distanceFromUnscared).

    * for ghostBonus:
    When Pacman eats the big dot, Pacman has the ability to eat ghosts.
    Get distanceFromScared with manhattan distance between ghost and pacman, and scaredTimer.
    Adapt the score by adding ghostBonus, which is the sum of (200 / square of distanceFromScared).

    * for foodBonus:
    Since the goal for Pacman is to eat all dots, dots can be regard as bonus.
    Get manhattanNearestFood with manhattan distance between dots and pacman position, and all the remained dots.
    Adapt the score by adding foodBonus, which is the sun of (10 / distance in manhattanNearestFood).

    Therefore, score = originalScore - ghostPenalty + ghostBonus + foodBonus.
    """
    # Begin your code (Part 4)
    state = currentGameState
    currentScore = state.getScore()
    if state.isWin() or state.isLose():
        return currentScore
    position = state.getPacmanPosition()
    ghosts = state.getGhostStates()

    ghostDistances = [manhattanDistance(position, ghost.configuration.pos) for ghost in ghosts]
    scaredTimers = [ghost.scaredTimer for ghost in ghosts]
    distFromUnscared = [dist for dist, timer in zip(ghostDistances, scaredTimers) if timer == 0]
    distFromScared = [dist for dist, timer in zip(ghostDistances, scaredTimers) if timer > 2]
    ghostPenalty = sum((300 / dist ** 2 for dist in distFromUnscared), 0)
    ghostBonus = sum((200 / dist for dist in distFromScared), 0)

    foods = state.getFood().asList()
    manhattanDistances = [(manhattanDistance(position, food), food) for food in foods]
    manhattanNearestFood = [food for dist, food in sorted(manhattanDistances)[:5]]
    foodBonus = sum(10 / d for d, f in manhattanNearestFood)

    score = currentScore - ghostPenalty + ghostBonus + foodBonus # + capsuleBonus
    return score
    # End your code (Part 4)

# Abbreviation
better = betterEvaluationFunction
