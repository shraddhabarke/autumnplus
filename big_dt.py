from read_trace import read_trace, Transition
from smt_encoding import solve as smt_solve
from dt_inference import infer as dt_infer

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

    return output_dt, next_state_dt

if __name__ == '__main__':
    trace = read_trace('gravity_i', 5)[2:]
    states = smt_solve(trace)
    num_states = max(states) + 1  # Hacky
    output_dt, next_state_dt = big_dt(trace, num_states, states)
    print(f'There are {num_states} states')
    print('Output decision tree:')
    output_dt.debug_print(prefix='  ')
    print('New state decision tree:')
    next_state_dt.debug_print(prefix='  ')


