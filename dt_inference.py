'''
Decision tree inference, using heuristic search with an information gain
heuristic, as described in:
 - Rajeev Alur, Arjun Radhakrishna, Abhishek Udupa: Scaling Enumerative Program
   Synthesis via Divide and Conquer. TACAS'17.
   https://arjunradhakrishna.github.io/publications/tacas2017.pdf
 - Quinlan: Induction of decision trees. Machine Learning, 1986.
   https://link.springer.com/article/10.1007/BF00116251

Input:
 - a description of the predicates {name: number of possible values}
 - a list of obs ({predicate name: value}, {set of possible results})

'''

from typing import *
from dataclasses import dataclass
from collections import Counter
import functools, operator, math

class DT:
    def debug_print(self, prefix=''): raise NotImplementedError
    def size(self) -> int: raise NotImplementedError

@dataclass
class DTBranch:
    pred: str
    children: list[DT]

    def debug_print(self, prefix=''):
        print(f'{prefix}Split on {self.pred}:')
        for c in self.children:
            c.debug_print(prefix=prefix+'  ')
    def size(self) -> int: return 1 + sum(c.size() for c in self.children)

@dataclass
class DTLeaf:
    value: set[str]

    def debug_print(self, prefix=''):
        if len(self.value) == 1:
            val, = self.value
            print(f'{prefix}Output {val}')
        else:
            print(f'{prefix}Output any from {self.value}')
    def size(self) -> int: return 1

@dataclass
class DTUnreachable:
    def debug_print(self, prefix=''):
        print(f'{prefix}UNREACHABLE (no data)')
    def size(self) -> int: return 0

ObsTable = list[tuple[dict[str, int], set[str]]]

def infer(preds: dict[str, int], obs: ObsTable) -> DT:
    # Base case: no data
    if len(obs) == 0:
        return DTUnreachable()

    # Base case: look for a common output value
    common = functools.reduce(operator.and_, (out for data, out in obs))
    if len(common) > 0:
        return DTLeaf(common)

    # Pick a predicate: information gain heuristic
    pred, num_values = min(preds.items(),
            key=lambda p: sum(entropy_eusolver(part) for part in split(obs, *p)))

    # Split on the predicate and recur
    preds = preds.copy()
    del preds[pred]
    return DTBranch(
            pred=pred,
            children=[infer(preds, part) for part in split(obs, pred, num_values)])

def split(obs: ObsTable, pred: str, num_values: int) -> list[ObsTable]:
    partition = [[] for i in range(num_values)]
    for data, out in obs:
        partition[data[pred]].append((data, out))
    return partition

def entropy_simple(obs: ObsTable) -> float:
    '''The entropy of a set of observations.'''
    count = Counter()
    for data, out in obs:
        for t in out:
            count[t] += 1/len(out)
    N = len(obs)
    return N * sum(- n/N * math.log(n/N) for n in count.values())

def entropy_eusolver(obs: ObsTable) -> float:
    # This is (equivalent to) what EUSolver does
    # I don't get it but it works
    cover_size = Counter()
    for data, out in obs:
        for t in out:
            cover_size[t] += 1
    count = Counter()
    for data, out in obs:
        denom = sum(cover_size[t] for t in out)
        for t in out:
            count[t] += cover_size[t] / denom
    total = len(obs)
    return total * sum(- n/total * math.log(n/total) for n in count.values())


if __name__ == '__main__':
    # Example from the 1986 paper
    preds = {'outlook': 3, 'temp': 3, 'humidity': 2, 'windy': 2}
    obs = [(dict(zip(['outlook', 'temp', 'humidity', 'windy'], data)), {out})
           for *data, out in [
               (0, 2, 1, 0, 'N'),
               (0, 2, 1, 1, 'N'),
               (1, 2, 1, 0, 'P'),
               (2, 1, 1, 0, 'P'),
               (2, 0, 0, 0, 'P'),
               (2, 0, 0, 1, 'N'),
               (1, 0, 0, 1, 'P'),
               (0, 1, 1, 0, 'N'),
               (0, 0, 0, 0, 'P'),
               (2, 1, 0, 0, 'P'),
               (0, 1, 0, 1, 'P'),
               (1, 1, 1, 1, 'P'),
               (1, 2, 0, 0, 'P'),
               (2, 1, 1, 1, 'N'),
           ]]
    dt = infer(preds, obs)
    print(dt)

