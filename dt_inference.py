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
from dataclasses import dataclass, field
from collections import Counter, defaultdict
import functools, itertools, operator, math, copy

class DT:
    '''The size of a decision tree is the number of its leaves'''
    size: int

    def debug_print(self, prefix=''): raise NotImplementedError

@dataclass(frozen=True)
class DTBranch(DT):
    pred: str
    children: tuple[DT, ...]
    size: int = field(init=False)

    def __post_init__(self):
        object.__setattr__(self, 'size', sum(c.size for c in self.children))

    def debug_print(self, prefix=''):
        print(f'{prefix}Split on {self.pred}:')
        for c in self.children:
            c.debug_print(prefix=prefix+'  ')

@dataclass(frozen=True)
class DTLeaf(DT):
    value: set[str]
    size: int = 1

    def debug_print(self, prefix=''):
        if len(self.value) == 1:
            val, = self.value
            print(f'{prefix}Output {val}')
        else:
            print(f'{prefix}Output any from {self.value}')

@dataclass(frozen=True)
class DTUnreachable(DT):
    size: int = 1

    def debug_print(self, prefix=''):
        print(f'{prefix}UNREACHABLE (no data)')

ObsTable = list[tuple[dict[str, int], set[str]]]

def heuristic_infer(preds: dict[str, int], obs: ObsTable) -> DT:
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
            children=tuple(heuristic_infer(preds, part)
                           for part in split(obs, pred, num_values)))

def split(obs: ObsTable, pred: str, num_values: int) -> list[ObsTable]:
    partition = [[] for i in range(num_values)]
    for data, out in obs:
        partition[data[pred]].append((data, out))
    return partition

def entropy_simple(obs: ObsTable) -> float:
    '''
    The entropy of a set of observations, using a simple probability
    distribution
    '''
    count = Counter()
    for data, out in obs:
        for t in out:
            count[t] += 1/len(out)
    N = len(obs)
    return N * sum(- n/N * math.log(n/N) for n in count.values())

def entropy_eusolver(obs: ObsTable) -> float:
    '''
    The entropy of a set of observations, using a more complicated probability
    distribution

    This is (equivalent to) what EUSolver does. I don't get it but it works
    '''
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


####### Branch and bound

def lower_bound_on_dt_size(obs: ObsTable, lo: int, hi: int) -> int:
    '''
    Return a lower bound on the size of a decision tree needed to cover
    obs[lo:hi]
    '''
    # The size of a decision tree is the number of leaf nodes
    # This is lower bounded by the size of a minimal covering set of labels
    # So we need to compute a set cover lower bound

    return set_cover_lower_bound(obs[i][1] for i in range(lo, hi))

def set_cover_lower_bound(items: Iterable[set[str]]) -> int:
    '''
    Set cover: at least one label (string) from each set must be selected.
    What's the fewest number of labels you can select?

    This function returns a *lower bound* on the number of labels needed.
    '''
    # Set Cover Lower Bound:
    # pick a set of representative examples s.t. each label only works for at
    # most one representative example
    # Then the minimal covering set is at least as big as the set of
    # representative examples
    # Greedy search, look at examples that correspond to a small number of
    # labels first
    #
    # This is a simplification of the LP-relaxation dual-fitting-based lower
    # bound algorithm from Vazirani, "Approximation algorithms", ch. 13

    size_to_labels = defaultdict(lambda: [])
    for labels in items:
        size_to_labels[len(labels)].append(labels)

    labels_used = set()
    num_representatives = 0
    for size in sorted(size_to_labels.keys()):
        for labels in size_to_labels[size]:
            if all(l not in labels_used for l in labels):
                num_representatives += 1
                labels_used.update(labels)

    return num_representatives

def branch_and_bound(
        preds: dict[str, int], obs: ObsTable, upper_bound = math.inf
        ) -> list[DT]:
    pred_list = list(preds.keys())

    def go(depth: int, upper_bound: int, lo: int, hi: int) -> list[DT]:
        '''
        Internal recursive helper function. Find all decision trees with:
         - size at most upper_bound
         - using predicates from pred_list[depth:]
         - matching observations in obs[lo:hi]

        The recursive algorithm modifies (reorders) both obs and pred_list
        in-place.
        '''
        # Base case: upper bound is too low
        if upper_bound <= 0:
            return []

        # Base case: no data
        if lo == hi:
            return [DTUnreachable()]

        # Base case: look for a common output value
        common = functools.reduce(operator.and_, (obs[i][1] for i in range(lo, hi)))
        if len(common) > 0:
            return [DTLeaf(common)]

        dts_size_equal_upper_bound: list[DT] = []

        # Pick a predicate from predicate list
        for pred_index in range(depth, len(pred_list)):
            pred = pred_list[pred_index]
            pred_list[depth], pred_list[pred_index] = pred, pred_list[depth]

            # Sort the range on that predicate
            endpts: list[int] = []
            sort_range_pred_vals(obs, lo, hi, pred, 0, preds[pred], endpts)

            # Compute lower bounds
            lo_ = lo
            bounds: list[int] = []
            for hi_ in endpts:
                bounds.append(lower_bound_on_dt_size(obs, lo_, hi_))
                lo_ = hi_
            assert lo_ == hi

            # Recurse on each range
            size_so_far = 0
            solns: list[list[DT]] = []
            lo_ = lo
            for i, hi_ in enumerate(endpts):
                result = go(depth + 1, upper_bound - size_so_far - sum(bounds[i+1:]), lo_, hi_)
                if len(result) == 0:
                    solns = None
                    break
                size_so_far += result[0].size
                solns.append(result)
                lo_ = hi_

            # Assemble into overall solutions
            if solns is not None:
                assert lo_ == hi
                assert size_so_far <= upper_bound

                if size_so_far < upper_bound:
                    dts_size_equal_upper_bound.clear()
                    upper_bound = size_so_far

                dts_size_equal_upper_bound.extend(
                        DTBranch(pred, children)
                        for children in itertools.product(*solns))

            pred_list[depth], pred_list[pred_index] = pred_list[pred_index], pred

        return dts_size_equal_upper_bound

    upper_bound = min(upper_bound, heuristic_infer(preds, obs).size)
    return go(0, upper_bound, 0, len(obs))

def sort_range_pred_vals(
        obs: ObsTable, lo: int, hi: int, pred: str, pred_lo: int, pred_hi: int,
        endpts: list[int]
        ):
    # Quicksort on predicate value
    # Asymptotically, this is only O(n * log(number of predicate values)), not
    # O(n*log(n)), but it's in Python so it's probably super slow anyways
    assert hi >= lo
    assert pred_hi > pred_lo
    if pred_hi - pred_lo <= 1:
        endpts.append(hi)
        return
    pred_mid = (pred_hi + pred_lo) // 2

    # Partition on lambda x: x < pred_mid
    a = lo
    b = hi
    while a < b:
        while a < b and obs[a][0][pred] < pred_mid:
            a += 1
        while a < b and obs[b-1][0][pred] >= pred_mid:
            b -= 1
        if a < b:
            obs[a], obs[b-1] = obs[b-1], obs[a]
            a += 1
            b -= 1
    mid = a
    assert a == mid == b
    sort_range_pred_vals(obs, lo, mid, pred, pred_lo, pred_mid, endpts)
    sort_range_pred_vals(obs, mid, hi, pred, pred_mid, pred_hi, endpts)



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
    dt = heuristic_infer(preds, obs)
    dt.debug_print()

    others = branch_and_bound(preds, obs)
    for dt in others:
        dt.debug_print()

