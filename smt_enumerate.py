from typing import *
from dataclasses import dataclass
import functools, itertools, math

from read_trace import read_trace, Transition
import dt_inference
from dt_inference import DT, Observation

from z3 import *


###### SMT enumeration

def fixed_num_states_encode(
        trace: list[Transition], max_states: int
        ) -> Tuple[Solver, list[Int]]:
    '''
    Returns a solver and a list of N+1 SMT variables for the integer state IDs
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

def smt_enumerate(trace: list[Transition]) -> Iterator[list[int]]:
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


###### Constructing an overall "big decision tree"

@dataclass
class BigDT:
    num_states: int
    output_dt: DT
    next_state_dt: DT

    def size(self) -> int:
        return self.output_dt.size + self.next_state_dt.size
    def debug_print(self, prefix=''):
        print(f'{prefix}There are {self.num_states} states and {self.size()} decision tree nodes')
        print(f'{prefix}Output decision tree:')
        self.output_dt.debug_print(prefix=prefix+'  ')
        print(f'{prefix}Next state decision tree:')
        self.next_state_dt.debug_print(prefix=prefix+'  ')

def big_dt(
        trace: list[Transition],
        num_states: int,
        states: list[int],
        upper_bound = math.inf):
    preds = {f'pred_{i}': 2 for i in range(len(trace[0].bits))}
    preds['state'] = num_states

    input_data = \
        [ {f'pred_{i}': p for i, p in enumerate(t.bits)} | {'state': states[i]}
        for i, t in enumerate(trace) ]

    # First DT learning problem: output
    # The next state DT is lower-bounded by the number of states, plus one for
    # staying put
    # FIXME (Mark 9/2/23): am I sure that there will always necessarily be a
    # "stay in the same state" output in the next state DT?
    output_upper_bound = upper_bound - num_states - 1
    data = [Observation(i, data, t.possible_actions)
            for i, (data, t) in enumerate(zip(input_data, trace))]

    output_dt = dt_inference.branch_and_bound(preds, data, output_upper_bound)
    if output_dt is None: return None

    # Second DT learning problem: next state
    next_state_upper_bound = upper_bound - output_dt.size
    data = [Observation(i, data, {states[i+1], 'stay put'} if states[i] == states[i+1] else {states[i+1]})
            for i, data in enumerate(input_data)]

    next_state_dt = dt_inference.branch_and_bound(preds, data, next_state_upper_bound)
    if next_state_dt is None: return None

    return BigDT(num_states, output_dt, next_state_dt)


###### Putting it all together

# TODO


if __name__ == '__main__':
    import time
    trace = read_trace('gravity_i', 5)[2:]

    state_assignments = list(smt_enumerate(trace))

    overall_start = time.time()
    COUNT = 100
    for i in range(COUNT):
        start = time.time()
        best = None
        best_size = math.inf
        for s in state_assignments:
            num_states = max(s) + 1

            dt = big_dt(trace, num_states, s, upper_bound = best_size)
            if dt is None: continue
            assert dt.size() < best_size
            best = dt
            best_size = dt.size()
        duration = time.time() - start
        print(f'This iteration took {duration} secs')
    print(f'The average time was {(time.time() - overall_start)/COUNT}')

    best.debug_print()

