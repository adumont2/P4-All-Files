# analysis.py
# ----------------
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


######################
# ANALYSIS QUESTIONS #
######################

# Set the given parameters to obtain the specified policies through
# value iteration.

def question2():
    """
      Change 'NOT POSSIBLE' to the parameter 'noise' or 'discount'
      and set the corresponding value to whatever you think is an
      appropriate value.
    """
    return 'discount', 0.99

def question3a():
    """
      Prefer the close exit (+1), risking the cliff (-10).
    """
    return 'discount', 0.1, 'noise', 0.0, 'livingReward', 0.0

def question3b():
    """
      Prefer the close exit (+1), but avoiding the cliff (-10).
    """
    return 'discount', 0.1, 'noise', 0.2, 'livingReward', 0.0

def question3c():
    """
      Prefer the distant exit (+10), risking the cliff (-10).
    """
    return 'discount', 0.9, 'noise', 0.0, 'livingReward', -0.1

def question3d():
    """
      Prefer the distant exit (+10), avoiding the cliff (-10).
    """
    return 'discount', 0.9, 'noise', 0.2, 'livingReward', -0.1

def question3e():
    """
      Avoid both exits and the cliff (so an episode should never terminate).
    """
    return 'discount', 0.9, 'noise', 0.0, 'livingReward', 1.0

def question6():
    """
      Used to be question 4.
      Change 'NOT POSSIBLE' to the parameter 'epsilon' or 'gamma' or 'alpha'
      and set the value.
    """
    return 'NOT POSSIBLE', 0.0

if __name__ == '__main__':
    print('Answers to analysis questions:')
    import analysis
    for q in [q for q in dir(analysis) if q.startswith('question')]:
        response = getattr(analysis, q)()
        print('  Question %s:\t%s' % (q, str(response)))
