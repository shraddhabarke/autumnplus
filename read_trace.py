from typing import *
import csv, ast
from dataclasses import dataclass

@dataclass
class Transition:
    possible_actions: set[str]
    bits: list[bool]

def read_trace(name: str, obj_id: int) -> list[Transition]:
    '''
    Extract a single trace from a CSV file into a list of transitions.

    Example:
    >>> # get the traces for the three particles in gravity_i
    >>> traces = [read_trace('gravity_i', obj_id) for obj_id in (5, 6, 7)]
    '''
    class dialect(csv.excel): delimiter = '|'
    update = [l for l in csv.reader(open(f'traces/{name}updates.csv'), dialect)]
    actions = [set(ast.literal_eval(x.removeprefix('Any'))) for x in update[obj_id - 1]]
    def process_pred(line):
        name, vals = line
        vals = ast.literal_eval(vals)
        if isinstance(vals, dict) and str(obj_id) in vals.keys():
            vals = vals[str(obj_id)]
        elif isinstance(vals, dict) and str(obj_id) not in vals.keys():
            return None
        return [bool(x) for x in vals]

    preds = [process_pred(l) for l in csv.reader(open(f'traces/{name}predicate.csv'), dialect) if process_pred(l) is not None]
    return list(map(lambda actions, *bits: Transition(actions, list(bits)), actions, *preds))

print(read_trace('paint', 7))