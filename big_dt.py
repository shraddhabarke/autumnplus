from read_trace import read_trace, Transition
from smt_encoding import solve as smt_solve
from dt_inference import DT, Observation, branch_and_bound as dt_infer
from dataclasses import dataclass
import math

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

    output_dt = dt_infer(preds, data, output_upper_bound)
    if output_dt is None: return None

    # Second DT learning problem: next state
    next_state_upper_bound = upper_bound - output_dt.size
    data = [Observation(i, data, {states[i+1], 'stay put'} if states[i] == states[i+1] else {states[i+1]})
            for i, data in enumerate(input_data)]

    next_state_dt = dt_infer(preds, data, next_state_upper_bound)
    if next_state_dt is None: return None

    return BigDT(num_states, output_dt, next_state_dt)

if __name__ == '__main__':
    trace = read_trace('gravity_i', 5)[2:]
    states = smt_solve(trace)
    num_states = max(states) + 1  # Hacky
    all_dts = big_dt(trace, num_states, states)
    print(f'Overall, there are {len(all_dts)} many minimal decision trees')
    for dt in all_dts:
        print(60*'-')
        dt.debug_print()

