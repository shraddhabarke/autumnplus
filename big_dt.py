from read_trace import read_trace, Transition
from smt_encoding import solve as smt_solve
from dt_inference import DT, infer as dt_infer
from dataclasses import dataclass

@dataclass
class BigDT:
    num_states: int
    output_dt: DT
    next_state_dt: DT

    def size(self) -> int:
        return self.output_dt.size() + self.next_state_dt.size()
    def debug_print(self):
        print(f'There are {self.num_states} states and {self.size()} decision tree nodes')
        print('Output decision tree:')
        self.output_dt.debug_print(prefix='  ')
        print('Next state decision tree:')
        self.next_state_dt.debug_print(prefix='  ')

def big_dt(trace: list[Transition], num_states: int, states: list[int]):
    preds = {f'pred_{i}': 2 for i in range(len(trace[0].bits))}
    preds['state'] = num_states

    input_data = \
        [ {f'pred_{i}': p for i, p in enumerate(t.bits)} | {'state': states[i]}
        for i, t in enumerate(trace) ]

    # First DT learning problem: output
    data = list(zip(input_data, (t.possible_actions for t in trace)))

    output_dt = dt_infer(preds, data)

    # Second DT learning problem: next state
    data = list(zip(input_data,
        ({states[i+1], 'stay put'} if states[i] == states[i+1] else {states[i+1]}
         for i in range(len(trace)))))

    next_state_dt = dt_infer(preds, data)

    return BigDT(num_states, output_dt, next_state_dt)

if __name__ == '__main__':
    trace = read_trace('gravity_i', 5)[2:]
    states = smt_solve(trace)
    num_states = max(states) + 1  # Hacky
    dt = big_dt(trace, num_states, states)
    dt.debug_print()

