'''
Baseline SMT-based DFA synthesis for benchmarks in the format of Oliveira et al,
2001
Example file: oliveira-bench-randm08.02.02.06.020_0030.05.aba.beta
The only source I found for these benchmarks is to download the artifact for the
Lbox paper and extract them from the docker image
'''

from typing import *
import functools
import itertools
import dataclasses
import time
import sys

from z3 import *

@dataclasses.dataclass
class PTA:
    value: Optional[bool]
    children: dict[str, 'PTA']
    state: Optional[Int] = None

    @classmethod
    def empty(cls) -> 'PTA':
        return PTA(None, {})

    @classmethod
    def from_file(cls, file: Iterable[str]) -> 'PTA':
        pta = PTA.empty()
        for line in file:
            line = line.rstrip()
            value = {'+': True, '-': False}[line[-1]]

            # This is not the only format but it's the only one I'm supporting
            # right now
            assert line[-2] == ','  
            example = line[:-2]
            assert all(c in '01' for c in example)
            pta.add(example, value)
        return pta

    def add(self, example: str, value: bool):
        if example == '':
            assert self.value in {value, None}
            self.value = value
            return
        if example[0] not in self.children:
            self.children[example[0]] = PTA.empty()
        self.children[example[0]].add(example[1:], value)

    def all_nodes(self) -> Iterable['PTA']:
        yield self
        for child in self.children.values():
            yield from child.all_nodes()


@dataclasses.dataclass
class State:
    state_id: int
    accepting: bool
    trans: dict[str, int]

class Timer:
    def __init__(self, timer_name):
        self.timer_name = timer_name
        self.start = self.time = time.perf_counter()
    def done(self, name: str):
        end = time.perf_counter()
        print(f'TIMER {self.timer_name}: {name} took {end-self.time} seconds',
              file=sys.stderr)
        self.time = end
    def __del__(self):
        end = time.perf_counter()
        print(f'TIMER {self.timer_name}: Overall took {end-self.start} seconds',
              file=sys.stderr)

def fixed_num_states_encode(
        pta: PTA, max_states: int
        ) -> Tuple[Solver, Optional[list[State]]]:
    s = Solver()

    timer = Timer(f'{max_states} states')

    # Define the states
    states = []
    for n in pta.all_nodes():
        n.state = Int(f'state_{len(states)}')
        states.append(n.state)
        s.add(0 <= n.state, n.state < max_states)
    timer.done('Defining states')

    # Symmetry-breaking
    s.add(states[0] == 0)
    # Very oddly, it takes ages to generate all these formulas, and only
    # generating some of them (controlled by MAX_CLAUSE_SIZE) makes it miles
    # faster
    # The really odd part is that it doesn't take long at all to solve it --
    # just *generating* the formulas is slow. IDEK ¯\_(ツ)_/¯
    #
    # I think this is probably some performance bug with the Python Z3 bindings
    MAX_CLAUSE_SIZE = 30 # IDK
    for i, j in itertools.product(range(1, min(len(states), MAX_CLAUSE_SIZE)), range(2, max_states)):
        s.add(Not(And(states[i] == j, *(s != j-1 for s in states[:i]))))
    timer.done('Symmetry breaking')

    # Define the outputs
    outputs = []
    for n in pta.all_nodes():
        outputs.append(Bool(f'out_{len(actions)}') if n.value is None else n.value)
    timer.done('Defining outputs')

    # Define the transition and output functions
    St = IntSort()
    Σ = IntSort()
    transition_fn = Function('T', St, Σ, St)
    output_fn = Function('O', St, BoolSort())
    for state, output in zip(states, outputs):
        s.add(output_fn(state) == output)
    for n in pta.all_nodes():
        for c, m in n.children.items():
            s.add(transition_fn(n.state, ord(c)) == m.state)
    timer.done('Defining DFA functions')

    # Check the model
    result = s.check()
    timer.done('Checking the model')
    if result == unsat:
        return s, None
    m = s.model()
    return s, [
            State(
                state_id = i,
                accepting = m.eval(output_fn(i)),
                trans = {c: m.eval(transition_fn(i, ord(c))) for c in '01'},
            )
            for i in range(max_states)
        ]


def solve(pta: PTA) -> list[State]:
    for i in itertools.count(1):
        s, states = fixed_num_states_encode(pta, i)
        if states is not None: return states

def dot(states: list[State]) -> str:
    out = 'digraph sfdjlksdfljksdfjlkdfs {\n'
    for s in states:
        for char, next_state in s.trans.items():
            out += f'  {s.state_id} -> {next_state} '
            out += f'[label="{char}"]\n'
    out += '}'
    return out

if __name__ == '__main__':
    pta = PTA.from_file(open('oliveira-bench-randm08.02.02.06.020_0030.05.aba.beta'))
    timer = Timer('Entire solve')
    states = solve(pta)
    del timer
    print(dot(states))

