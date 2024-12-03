# search.py
# ---------
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


"""
In search.py, you will implement generic search algorithms which are called by
Pacman agents (in searchAgents.py).
"""

import util
from game import Directions
from typing import List

class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the search problem.
        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state.
        """
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples, (successor,
        action, stepCost), where 'successor' is a successor to the current
        state, 'action' is the action required to get there, and 'stepCost' is
        the incremental cost of expanding to that successor.
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves.
        """
        util.raiseNotDefined()




def tinyMazeSearch(problem: SearchProblem) -> List[Directions]:
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    s = Directions.SOUTH
    w = Directions.WEST
    return  [s, s, w, s, w, w, s, w]

def depthFirstSearch(problem: SearchProblem) -> List[Directions]:
    """
    Search the deepest nodes in the search tree first.

    Your search algorithm needs to return a list of actions that reaches the
    goal. Make sure to implement a graph search algorithm.

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:

    print("Start:", problem.getStartState())
    print("Is the start a goal?", problem.isGoalState(problem.getStartState()))
    print("Start's successors:", problem.getSuccessors(problem.getStartState()))
    """
    "*** YOUR CODE HERE ***"

    actions = util.Stack()
    best = False

    if not best:
        reached = set()

        def dfs(state):
            reached.add(state)
            if problem.isGoalState(state):
                return actions.list
            for (next_state, dir, _) in problem.getSuccessors(state):
                if next_state not in reached:
                    actions.push(dir)
                    exp = dfs(next_state)
                    if exp:
                        return exp
                    _ = actions.pop()
            return None
        
        return dfs(problem.getStartState())
    
    else:
        tot_cost = {problem.getStartState(): 0}
        best_path = []

        def dfs(state):
            if problem.isGoalState(state):
                nonlocal best_path
                best_path = actions.list
            for (next_state, dir, action_cost) in problem.getSuccessors(state):
                if next_state not in tot_cost or tot_cost[next_state] > tot_cost[state] + action_cost:
                    actions.push(dir)
                    tot_cost[next_state] = tot_cost[state] + action_cost
                    dfs(next_state)
                    _ = actions.pop()
        
        dfs(problem.getStartState())
        return best_path
    


def breadthFirstSearch(problem: SearchProblem) -> List[Directions]:
    """Search the shallowest nodes in the search tree first."""
    "*** YOUR CODE HERE ***"
    q = util.Queue()
    q.push((problem.getStartState(), []))
    visited = set([problem.getStartState()])

    i = 0
    while len(q.list) > 0:
        # print(i)
        i += 1
        node = q.pop()
        if problem.isGoalState(node[0]):
            return node[1]
        for (next_state, dir, _) in problem.getSuccessors(node[0]):
            if next_state not in visited:
                visited.add(next_state)

                next_path = node[1].copy()
                next_path.append(dir)
                q.push((next_state,next_path))

    print("No RESULT FOUND")
    return None


def uniformCostSearch(problem: SearchProblem) -> List[Directions]:
    """Search the node of least total cost first."""
    "*** YOUR CODE HERE ***"
    visited = set()
    q = util.PriorityQueue()
    q.push(problem.getStartState(), 0)
    path = {problem.getStartState(): []}
    cost_dict = {problem.getStartState(): 0}
    while q:
        state = q.pop()
        visited.add(state)
        if problem.isGoalState(state):
            return path[state]
        for(next_state, dir, cost) in problem.getSuccessors(state):
            if next_state not in visited:
                if next_state not in cost_dict or cost_dict[next_state] > cost_dict[state] + cost:
                    cost_dict[next_state] = cost_dict[state] + cost
                    path[next_state] = path[state].copy() + [dir]
                q.update(next_state, cost_dict[next_state])
    return None


def nullHeuristic(state, problem=None) -> float:
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def WRONGaStarSearch(problem: SearchProblem, heuristic=nullHeuristic) -> List[Directions]:
    """Search the node that has the lowest combined cost and heuristic first."""
    "*** YOUR CODE HERE ***"
    for _ in range(5):
        print()
    visited = set()
    q = util.PriorityQueue()
    q.push(problem.getStartState(), heuristic(problem.getStartState(), problem))
    path = {problem.getStartState(): []}
    cost_dict = {problem.getStartState(): 0}
    while q:
        first = q.heap[0]
        state = q.pop()
        print("Popped", state, first[0])
        print(cost_dict)
        # print(path)
        visited.add(state)
        if problem.isGoalState(state):
            return path[state]
        for(next_state, dir, cost) in problem.getSuccessors(state):
            if next_state not in visited:
                if next_state not in cost_dict or cost_dict[next_state] > cost_dict[state] + cost:
                    print("update", state, next_state)
                    cost_dict[next_state] = cost_dict[state] + cost
                    path[next_state] = path[state].copy() + [dir]
                q.update(next_state, cost_dict[next_state] + heuristic(next_state, problem))
    return None

def aStarSearch(problem: SearchProblem, heuristic=nullHeuristic) -> List[Directions]:
    """Search the node that has the lowest combined cost and heuristic first."""
    "*** YOUR CODE HERE ***"
    q = util.PriorityQueue()
    q.push(problem.getStartState(), heuristic(problem.getStartState(), problem))
    path = {problem.getStartState(): []}
    cost_dict = {problem.getStartState(): 0}
    while q:
        state = q.pop()
        if problem.isGoalState(state):
            return path[state]
        for(next_state, dir, cost) in problem.getSuccessors(state):
            if next_state not in cost_dict or cost_dict[next_state] > cost_dict[state] + cost:
                    # print("update", state, next_state)
                    # print(f"From {cost_dict[next_state] if next_state in cost_dict else "none"} to {cost_dict[state] + cost}")
                    cost_dict[next_state] = cost_dict[state] + cost
                    path[next_state] = path[state].copy() + [dir]
                    q.update(next_state, cost_dict[next_state] + heuristic(next_state, problem))
    return None

# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
wastar = WRONGaStarSearch
