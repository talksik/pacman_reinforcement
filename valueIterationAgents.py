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


import mdp
import util

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

    def __init__(self, mdp, discount=0.9, iterations=100):
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
        self.values = util.Counter()  # A Counter is a dict with default 0
        self.runValueIteration()

    def runValueIteration(self):
        # Write value iteration code here
        "*** YOUR CODE HERE ***"
        INF, NEG_INF = float("inf"), -float("inf")

        # Run through iterations
        for i in range(self.iterations):
            # copy function defined?
            policy = self.values.copy()

            # MDP states
            mdp_states = self.mdp.getStates()
            for curr_state in mdp_states:
                # curr state is exit
                if not self.mdp.isTerminal(curr_state):
                    options_actions = self.mdp.getPossibleActions(curr_state)
                    optimal = max([self.getQValue(curr_state, x)
                                   for x in options_actions])

                    # add optimal to the policy
                    policy[curr_state] = optimal
            # Update the new best policy
            self.values = policy

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
        curr_val = 0

        possible = self.mdp.getTransitionStatesAndProbs(state, action)
        for new_state, prob in possible:
            r = self.mdp.getReward(state, action, new_state)

            val = self.values[new_state]
            curr_val = curr_val + prob * ((self.discount * val) + r)

        return curr_val

    def computeActionFromValues(self, state):
        """
          The policy is the best action in the given state
          according to the values currently stored in self.values.

          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.
        """
        "*** YOUR CODE HERE ***"
        # end iteration
        if self.mdp.isTerminal(state):
            return None
        curr_val, optimal_action = -float("inf"), ''

        for action in self.mdp.getPossibleActions(state):
            curr_qval = self.computeQValueFromValues(state, action)
            # update if better
            if curr_qval >= curr_val:
                curr_val = curr_qval
                optimal_action = action

        return optimal_action

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

    def __init__(self, mdp, discount=0.9, iterations=1000):
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
        INF, NEG_INF = float("inf"), -float("inf")

        # MDP states
        mdp_states = self.mdp.getStates()

        # Run through iterations
        for i in range(self.iterations):
            # copy function defined?
            curr_state = mdp_states[i % len(mdp_states)]

            # not iterating all actions this time
            if not self.mdp.isTerminal(curr_state):
                options_actions = self.mdp.getPossibleActions(curr_state)
                optimal = max([self.getQValue(curr_state, x)
                               for x in options_actions])

                # add optimal to the policy
                self.values[curr_state] = optimal


class PrioritizedSweepingValueIterationAgent(AsynchronousValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A PrioritizedSweepingValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs prioritized sweeping value iteration
        for a given number of iterations using the supplied parameters.
    """

    def __init__(self, mdp, discount=0.9, iterations=100, theta=1e-5):
        """
          Your prioritized sweeping value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy.
        """
        self.theta = theta
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):
        "*** YOUR CODE HERE ***"
        # initiliaze empty PQ
        # Use priority queue from utils for algorithm order
        hinge = util.PriorityQueue()

        dictPrev = {}
        mdp_states = self.mdp.getStates()

        # computing the predecssors for all states
        # For each non-terminal state, do:

        for curr_state in mdp_states:
            # exit the iteration
            if self.mdp.isTerminal(curr_state):
                continue

            options_actions = self.mdp.getPossibleActions(curr_state)
            for action in options_actions:

                all_transitions = self.mdp.getTransitionStatesAndProbs(
                    curr_state, action)
                for new_state, prob in all_transitions:

                    if new_state in dictPrev:
                        dictPrev[new_state].add(curr_state)

                    else:
                        dictPrev[new_state] = {curr_state}

        mdp_states = self.mdp.getStates()

        # Find the absolute value of the difference between the current value of s in self.values and the highest Q-value across all possible actions from s (this represents what the value should be); call this number diff. Do NOT update self.values[s] in this step.
        # Push s into the priority queue with priority -diff (note that this is negative). We use a negative because the priority queue is a min heap, but we want to prioritize updating states that have a higher error.
        for curr_state in mdp_states:
            if not self.mdp.isTerminal(curr_state):
                options_actions = self.mdp.getPossibleActions(curr_state)
                optimal = max([self.getQValue(curr_state, x)
                               for x in options_actions])
                # finding -diff
                diff = abs(optimal - self.values[curr_state])
                hinge.update(curr_state, - diff)

        # For iterations
        #         For iteration in 0, 1, 2, ..., self.iterations - 1, do:
        # If the priority queue is empty, then terminate.
        # Pop a state s off the priority queue.
        # Update s's value (if it is not a terminal state) in self.values.
        # For each predecessor p of s, do:
        # Find the absolute value of the difference between the current value of p in self.values and the highest Q-value across all possible actions from p (this represents what the value should be); call this number diff. Do NOT update self.values[p] in this step.
        # If diff > theta, push p into the priority queue with priority -diff (note that this is negative), as long as it does not already exist in the priority queue with equal or lower priority. As before, we use a negative because the priority queue is a min heap, but we want to prioritize updating states that have a higher error.
        for i in range(self.iterations):
            # no processing to do
            if hinge.isEmpty():
                break

            curr_state = hinge.pop()
            if not self.mdp.isTerminal(curr_state):
                options_actions = self.mdp.getPossibleActions(curr_state)
                optimal = max([self.getQValue(curr_state, x)
                               for x in options_actions])

                self.values[curr_state] = optimal

            for prev in dictPrev[curr_state]:
                if self.mdp.isTerminal(prev):
                    continue

                options_actions = self.mdp.getPossibleActions(curr_state)
                optimal = max([self.getQValue(curr_state, x)
                               for x in options_actions])
                diff = abs(optimal - self.values[prev])
                # difference large enough?
                if diff > self.theta:
                    hinge.update(prev, -diff)
