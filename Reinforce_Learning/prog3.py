import sys
import random


def chooseAction(actions, goals, s, count, total, M):
    # Choose the Action
    top = max(goals.values())
    bottom = min(goals.values())
    n = len(actions[s])
    c = 0
    p = []
    for i in range(n):
        if count[s, i] == 0:
            return i
        c += count[s, i]
        avg = total[s, i] / count[s, i]
        savg = 0.25 + 0.75 * (avg - bottom) / (top - bottom)
        up = savg ** (c / M)
        p.append(up)
    p[:] = [x / sum(p) for x in p]
    index = random.choices(actions[s], weights=p)[0]
    return index


def output(actions, Count, Total, i):
    # This function is to output in format
    print("After", i, "rounds")
    print("Count:")
    for state in actions.keys():
        for action in actions[state]:
            print("[%d,%d]=%d." % (state, action, Count[state, action]), end=" ")
        print()
    print()
    print("Total:")
    for state in actions.keys():
        for action in actions[state]:
            print("[%d,%d]=%d." % (state, action, Total[state, action]), end=" ")
        print()
    print()
    print("Best action:", end=" ")
    for state in actions.keys():
        largest = []
        for action in actions[state]:
            if Count[state, action] == 0:
                largest.append(0)
            else:
                largest.append(Total[state, action] / Count[state, action])
        if max(largest) == 0:
            print("%d:U." % (state), end=" ")
        else:
            print("%d:%d." % (state, largest.index(max(largest))), end=" ")
    print()
    print()


# input file
f = open(sys.argv[1], "r")

# Parameters
Param = list(map(int, f.readline().replace("\n", "").split(" ")))

# Terminal nodes
TermNodes = f.readline().replace("\n", "").split(" ")
goals = {}
for i in range(int(len(TermNodes) / 2)):
    goals[int(TermNodes[2 * i])] = int(TermNodes[2 * i + 1])

# Available actions and the transition matrix
Nodes = f.readlines()
actions = {}
nodes = {}
for node in Nodes:
    n = node.replace("\n", "").split(" ")
    foward = []
    prob = []
    for i in range(int((len(n) - 1) / 2)):
        foward.append(int(n[2 * i + 1]))
        prob.append(float(n[2 * i + 2]))
    state, action = n[0].split(":")
    state = int(state)
    if state not in actions:
        actions[state] = [int(action)]
    else:
        actions[state].append(int(action))
    nodes[n[0]] = (foward, prob)
f.close()

# Set up a data structure that records for each non-terminal state S and action A
Count = {}
Total = {}
for state in actions:
    for action in actions[state]:
        Count[state, action] = 0
        Total[state, action] = 0
r = 0

# training
while r < Param[2]:
    initial_state = random.randint(0, Param[0] - 1)
    state = initial_state
    count = set([])
    while state not in goals.keys():
        action = chooseAction(actions, goals, state, Count, Total, Param[-1])
        count.add((state, action))
        sa = str(state) + ":" + str(action)
        foward, prob = nodes[sa]
        state = random.choices(foward, weights=prob)[0]
    goal = goals[state]
    for step in count:
        Count[step] += 1
        Total[step] += goal
    r += 1

    # output
    if Param[-2] == 0:
        if r == Param[2]:
            output(actions, Count, Total, r)
    else:
        if r % Param[-2] == 0:
            output(actions, Count, Total, r)
