
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
        ) -> Tuple[Solver, Optional[list[State]]]:
    s = Solver()
    N = len(trace)

    # Define the states
    states = [Int(f'state_{i}') for i in range(N+1)]
    for st in states:
        s.add(0 <= st, st < max_states)

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
    Preds = BitVecSort(len(trace[0].bits))
    transition_fn = Function('T', Z, Preds, Z)
    output_fn = Function('O', Z, Preds, Z)
    for i in range(N):
        preds = functools.reduce(lambda x, y: 2*x + y, trace[i].bits)
        s.add(transition_fn(states[i], preds) == states[i+1])
        s.add(output_fn(states[i], preds) == actions[i])

    # Check the model
    if s.check() == unsat:
        return s, None
    m = s.model()
    result = [State(i, {}) for i in range(max_states)]
    s_val = lambda i: m[states[i]].as_long()
    for i, t in enumerate(trace):
        preds = tuple(t.bits)
        start = result[s_val(i)]
        if preds in start.trans:
            start.trans[preds][0].intersection_update(t.possible_actions)
        else:
            start.trans[preds] = t.possible_actions, s_val(i+1)
    return s, result


def solve(trace: list[Transition]) -> list[State]:
    for i in itertools.count():
        s, states = fixed_num_states_encode(trace, i)
        if states is not None: return states

def dot(states: list[State]) -> str:
    pred_memo = {}
    out = 'digraph sfdjlksdfljksdfjlkdfs {'
    for s in states:
        for pred, (r, next_state) in s.trans.items():
            if pred not in pred_memo: pred_memo[pred] = len(pred_memo)
            out += f'  {s.state_id} -> {next_state} '
            out += f'[label="pred_{pred_memo[pred]} / {str(r)}"]\n'
    out += '}'
    return out

if __name__ == '__main__':
    trace = read_trace('gravity_i', 5)[2:]
    states = solve(trace)
    print(dot(states))

