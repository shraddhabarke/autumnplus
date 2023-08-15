from typing import *
import functools
import itertools

from read_trace import read_trace, Transition
from big_dt import big_dt
from z3 import *

def fixed_num_states_encode(
        trace: list[Transition], max_states: int
        ) -> Tuple[Solver, Optional[list[int]]]:
    '''
    Returns a solver and a list of N+1 integer state IDs
    '''
    s = Solver()
    N = len(trace)

    # Define the states
    states = [Int(f'state_{i}') for i in range(N+1)]
    for st in states:
        s.add(0 <= st, st < max_states)  # size constraints, iteratively increases max_states if unsat

    # Symmetry-breaking
    s.add(states[0] == 0)
    MAX_SYMMETRY_BREAKING = 40
    for i, j in itertools.product(range(1, min(MAX_SYMMETRY_BREAKING, N+1)), range(1, max_states)):
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
    Preds = BitVecSort(len(trace[0].bits))
    transition_fn = Function('T', Z, Preds, Z)
    output_fn = Function('O', Z, Preds, Z)
    for i in range(N):
        preds = functools.reduce(lambda x, y: 2*x + y, trace[i].bits)
        s.add(transition_fn(states[i], preds) == states[i+1])
        s.add(output_fn(states[i], preds) == actions[i])

    # Extra constraint: when nothing at all happens, the state does not change
    null_event = 0  # Is this right? Seems like it
    for st in states:
        s.add(transition_fn(st, null_event) == st)

    return s, states

def solve_all(trace: list[Transition]) -> Iterator[list[int]]:
    '''
    An iterator of all minimal automata
    '''
    # Find the least number of states needed
    for max_states in itertools.count(1):
        s, states = fixed_num_states_encode(trace, max_states)
        if s.check() == sat:
            break

    print(f'There are {max_states} states')

    # Enumerate all models
    while s.check() == sat:
        model = s.model()
        assignment = [model[st].as_long() for st in states]
        yield assignment
        s.add(Or(*(st != val for st, val in zip(states, assignment))))

if __name__ == '__main__':
    trace = read_trace('gravity_i', 5)[2:]
    dt = min((big_dt(trace, max(s)+1, s) for s in solve_all(trace)),
             key=lambda dt: dt.size())

    dt.debug_print()

