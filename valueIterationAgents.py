# valueIterationAgents.py
# -----------------------
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


# valueIterationAgents.py
# -----------------------
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


import mdp, util

from learningAgents import ValueEstimationAgent
import collections

class ValueIterationAgent(ValueEstimationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A ValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100):
        """
          Your value iteration agent should take an mdp on
          construction, run the indicated number of iterations
          and then act according to the resulting policy.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state, action, nextState)
              mdp.isTerminal(state)
        """
        self.mdp = mdp
        self.discount = discount
        self.iterations = iterations
        self.values = util.Counter() # A Counter is a dict with default 0
        self.runValueIteration()

    def runValueIteration(self):
        # Write value iteration code here
        "*** YOUR CODE HERE ***"
        """
        This is the main loop. It runs for self.iterations.
        In each iteration, it computes a new set of values (k+1)
        using the values from the previous iteration (k).
        """
        for i in range(self.iterations):
            # Create a new Counter to store the V_{k+1} values.
            # We must use the V_k values (self.values) for all
            # calculations within this iteration.
            newValues = util.Counter()

            # Iterate over all states in the MDP
            for state in self.mdp.getStates():
                # If it's a terminal state, its value is 0
                if self.mdp.isTerminal(state):
                    newValues[state] = 0
                else:
                    # Find the max Q-value for this state
                    # by checking all possible actions.
                    possibleActions = self.mdp.getPossibleActions(state)
                    if not possibleActions:
                        # If no actions, value is 0 (or technically
                        # based on just reward, but 0 is fine here)
                        newValues[state] = 0
                    else:
                        # Find the action with the highest Q-value
                        maxQValue = -float('inf')
                        for action in possibleActions:
                            qValue = self.computeQValueFromValues(state, action)
                            if qValue > maxQValue:
                                maxQValue = qValue
                        
                        # The new value for the state is this max Q-value
                        newValues[state] = maxQValue

            # After checking all states, update self.values
            # to be the newValues we just computed.
            # This completes iteration k.
            self.values = newValues


    def getValue(self, state):
        """
          Return the value of the state (computed in __init__).
        """
        return self.values[state]


    def computeQValueFromValues(self, state, action):
        """
          Compute the Q-value of action in state from the
          value function stored in self.values.
        """
        "*** YOUR CODE HERE ***"
        """
        This implements:
          Q(s, a) = Sum_{s'} [ T(s, a, s') * (R(s, a, s') + gamma * V(s')) ]
        """
        qValue = 0
        
        # Get all possible (nextState, probability) pairs
        transitions = self.mdp.getTransitionStatesAndProbs(state, action)
        
        for nextState, prob in transitions:
            # Get the reward for this specific transition
            reward = self.mdp.getReward(state, action, nextState)
            
            # Get the value V_k(s') from our current value function
            futureValue = self.values[nextState]
            
            # Add this transition's contribution to the total Q-value
            qValue += prob * (reward + self.discount * futureValue)
            
        return qValue

    def computeActionFromValues(self, state):
        """
          The policy is the best action in the given state
          according to the values currently stored in self.values.

          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.
        """
        "*** YOUR CODE HERE ***"
        # If it's a terminal state, there is no action
        if self.mdp.isTerminal(state):
            return None

        possibleActions = self.mdp.getPossibleActions(state)
        
        # If there are no actions, return None
        if not possibleActions:
            return None

        # Use a Counter to easily find the argMax
        actionValues = util.Counter()
        
        # Calculate the Q-value for every possible action
        for action in possibleActions:
            actionValues[action] = self.computeQValueFromValues(state, action)
            
        # argMax() returns the key (action) with the highest value
        return actionValues.argMax()

    def getPolicy(self, state):
        return self.computeActionFromValues(state)

    def getAction(self, state):
        "Returns the policy at the state (no exploration)."
        return self.computeActionFromValues(state)

    def getQValue(self, state, action):
        return self.computeQValueFromValues(state, action)

class AsynchronousValueIterationAgent(ValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        An AsynchronousValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs cyclic value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 1000):
        """
          Your cyclic value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy. Each iteration
          updates the value of only one state, which cycles through
          the states list. If the chosen state is terminal, nothing
          happens in that iteration.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state)
              mdp.isTerminal(state)
        """
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):
        "*** YOUR CODE HERE ***"
        states = self.mdp.getStates()
        numStates = len(states)

        for i in range(self.iterations):
            # Get the state to update in this cycle
            state = states[i % numStates]

            # If it's not a terminal state, update its value
            if not self.mdp.isTerminal(state):
                possibleActions = self.mdp.getPossibleActions(state)
                if possibleActions:
                    # Find the max Q-value over all possible actions
                    # Note: This uses computeQValueFromValues, which
                    # reads from self.values. Since we update
                    # self.values in-place, this is an
                    # asynchronous update.
                    maxQValue = max([self.computeQValueFromValues(state, a) for a in possibleActions])
                    
                    # Update the value (in-place)
                    self.values[state] = maxQValue
                else:
                    # Non-terminal state with no actions, value is 0
                    self.values[state] = 0

class PrioritizedSweepingValueIterationAgent(AsynchronousValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A PrioritizedSweepingValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs prioritized sweeping value iteration
        for a given number of iterations using the supplied parameters.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100, theta = 1e-5):
        """
          Your prioritized sweeping value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy.
        """
        self.theta = theta
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):
        "*** YOUR CODE HERE ***"
        # 1. Compute predecessors for all states
        predecessors = collections.defaultdict(set)
        for state in self.mdp.getStates():
            if not self.mdp.isTerminal(state):
                for action in self.mdp.getPossibleActions(state):
                    for nextState, prob in self.mdp.getTransitionStatesAndProbs(state, action):
                        if prob > 0:
                            predecessors[nextState].add(state)

        # 2. Initialize an empty priority queue
        pq = util.PriorityQueue()

        # 3. Populate the priority queue
        for state in self.mdp.getStates():
            if not self.mdp.isTerminal(state):
                possibleActions = self.mdp.getPossibleActions(state)
                if possibleActions:
                    # Find current value and best possible new value
                    currentValue = self.values[state]
                    maxQValue = max([self.computeQValueFromValues(state, a) for a in possibleActions])
                    
                    # Calculate the difference
                    diff = abs(currentValue - maxQValue)
                    
                    # Push into the queue with negative diff as priority
                    # (since PriorityQueue is a min-heap)
                    pq.push(state, -diff)

        # 4. Main loop
        for i in range(self.iterations):
            # If queue is empty, we are done
            if pq.isEmpty():
                break
                
            # Pop the state with the highest priority (largest diff)
            s = pq.pop()

            # Update the value of state s
            if not self.mdp.isTerminal(s):
                possibleActions = self.mdp.getPossibleActions(s)
                if possibleActions:
                     maxQValue = max([self.computeQValueFromValues(s, a) for a in possibleActions])
                     self.values[s] = maxQValue
                else:
                     self.values[s] = 0

            # 5. Update priorities for all predecessors of s
            for p in predecessors[s]:
                # Find the new best Q-value for predecessor p
                possibleActions_p = self.mdp.getPossibleActions(p)
                if possibleActions_p:
                    currentValue_p = self.values[p]
                    maxQ_p = max([self.computeQValueFromValues(p, a) for a in possibleActions_p])
                    
                    diff_p = abs(currentValue_p - maxQ_p)
                    
                    # If the diff is greater than theta, update/push it
                    if diff_p > self.theta:
                        pq.update(p, -diff_p)

