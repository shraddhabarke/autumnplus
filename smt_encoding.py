
from typing import *
import functools
import itertools
import dataclasses

from read_trace import read_trace, Transition
from z3 import *

@dataclasses.dataclass
class State:
    state_id: int
    trans: dict[tuple[Bool, ...], Tuple[set[str], int]]

def fixed_num_states_encode(
        trace: list[Transition], max_states: int
        ) -> Tuple[Solver, Optional[list[int]]]:
    '''
    Returns a solver and a list of N+1 integer state IDs
    '''
    s = Optimize()
    N = len(trace)

    # Define the states
    states = [Int(f'state_{i}') for i in range(N+1)]
    for st in states:
        s.add(0 <= st, st < max_states)   # size constraints, iteratively increases max_states if unsat
    # Symmetry-breaking
    s.add(states[0] == 0)
    for i, j in itertools.product(range(1, N+1), range(1, max_states)):
        s.add(Not(And(states[i] == j, *(s != j-1 for s in states[:i]))))

    # Define the actions
    action_ids = {}
    for t in trace:
        for a in t.possible_actions:
            if a not in action_ids:
                action_ids[a] = len(action_ids)
    actions = [Int(f'action_{i}') for i in range(N)]
    for t, act in zip(trace, actions):
        s.add(Or(*(act == action_ids[a] for a in t.possible_actions)))

    # Define the transition and output functions
    Z = IntSort()
    predDict = {}
    Preds = BitVecSort(len(trace[0].bits))
    transition_fn = Function('T', Z, Preds, Z)
    output_fn = Function('O', Z, Preds, Z)
    for i in range(N):
        preds = functools.reduce(lambda x, y: 2*x + y, trace[i].bits)
        s.add(transition_fn(states[i], preds) == states[i+1])
        if preds not in predDict: predDict[preds] = [i]
        else: predDict[preds].append(i)
        s.add(output_fn(states[i], preds) == actions[i])

    # constraints to behave similarly under same predicates
    for v in predDict.values():
        for i in range(len(v)):
            for j in range(i + 1, len(v)):
                s.add_soft((states[v[i]] == states[v[i]+1]) == (states[v[j]] == states[v[j]+1]))

    # Check the model
    if s.check() == unsat:
        return s, None
    model = s.model()
    return s, [model[st].as_long() for st in states]

def solve(trace: list[Transition]) -> list[int]:
    for max_states in itertools.count(1):
        s, states = fixed_num_states_encode(trace, max_states)
        if states is not None: return states

def make_evidence_automaton(
        trace: list[Transition], num_states: int, states: list[int]
        ) -> list[State]:
    result = [State(i, {}) for i in range(num_states)]
    for i, t in enumerate(trace):
        preds = tuple(t.bits)
        start = result[states[i]]
        if preds in start.trans:
            start.trans[preds][0].intersection_update(t.possible_actions)
        else:
            start.trans[preds] = t.possible_actions, states[i+1]
    return result

def dot(evidence_automaton: list[State]) -> str:
    pred_memo = {}
    out = 'digraph sfdjlksdfljksdfjlkdfs {\n'
    for s in evidence_automaton:
        for pred, (r, next_state) in s.trans.items():
            if pred not in pred_memo: pred_memo[pred] = len(pred_memo)
            out += f'  {s.state_id} -> {next_state} '
            out += f'[label="pred_{pred_memo[pred]} / {str(r)}"]\n'
    out += '}'
    return out

if __name__ == '__main__':
    trace = read_trace('gravity_i', 5)[2:]
    states = solve(trace)
    num_states = max(states) + 1  # Hacky
    evidence_automaton = make_evidence_automaton(trace, num_states, states)
    print(dot(evidence_automaton))
