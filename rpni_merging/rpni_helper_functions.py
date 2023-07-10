import pickle
import queue
from functools import total_ordering
from typing import Set

@total_ordering
class RpniNode:
    __slots__ = ['output', 'children', 'prefix', "type"]

    def __init__(self, output=None, children=None, automaton_type='mealy'):
        if output is None and automaton_type == 'mealy':
            output = dict()
        if children is None:
            children = dict()
        self.output = output
        self.children = children
        self.prefix = ()
        self.type = automaton_type
        
    def __lt__(self, other):
        return (len(self.prefix), self.prefix) < (len(other.prefix), other.prefix)

    def __eq__(self, other):
        return self.prefix == other.prefix

    def get_all_nodes(self) -> Set['RpniNode']:
        qu = queue.Queue()
        qu.put(self)
        nodes = set()
        while not qu.empty():
            state = qu.get()
            nodes.add(state)
            for child in state.children.values():
                if child not in nodes:
                    qu.put(child)
        return nodes

    def to_automaton(self):
        nodes = self.get_all_nodes()
        nodes.remove(self)  # dunno whether order is preserved?
        nodes = [self] + list(nodes)
        return to_automaton(nodes, self.type)

def createPTA(data, automaton_type):
    data.sort(key=lambda x: len(x[0]))

    root_node = RpniNode(automaton_type=automaton_type) # initialize root node at the start of the PTA construction
    for seq, label in data:
        curr_node = root_node # for new sequence, reassign current node to be the root node

        for idx, symbol in enumerate(seq):
            if symbol not in curr_node.children.keys():
                node = RpniNode(automaton_type=automaton_type)
                node.prefix = curr_node.prefix + (symbol,)
                curr_node.children[symbol] = node

            if automaton_type == 'mealy' and idx == len(seq) - 1:
                if symbol not in curr_node.output:
                    curr_node.output[symbol] = label
                if curr_node.output[symbol] != label:
                    return None
            curr_node = curr_node.children[symbol]  # move ahead in the sequence by one symbol

    return root_node

def to_automaton(red, automaton_type):
    from MealyMachine import MealyMachine, MealyState

    state, automaton = MealyState, MealyMachine

    initial_state = None
    prefix_state_map = {}
    for i, r in enumerate(red):
        prefix_state_map[r.prefix] = state(f's{i}')
        if i == 0:
            initial_state = prefix_state_map[r.prefix]

    for r in red:
        for i, c in r.children.items():
            if automaton_type == 'moore' or automaton_type == 'dfa':
                prefix_state_map[r.prefix].transitions[i] = prefix_state_map[c.prefix]
            else:
                prefix_state_map[r.prefix].transitions[i] = prefix_state_map[c.prefix]
                prefix_state_map[r.prefix].output_fun[i] = r.output[i] if i in r.output else None

    return automaton(initial_state, list(prefix_state_map.values()))

def visualize_pta(root_node, path='pta.pdf'):
    from pydot import Dot, Node, Edge
    graph = Dot('fpta', graph_type='digraph')

    graph.add_node(Node(str(root_node.prefix), label=f'{root_node.output}'))

    queue = [root_node]
    visited = set()
    visited.add(root_node.prefix)
    while queue:
        curr = queue.pop(0)
        for i, c in curr.children.items():
            if c.prefix not in visited:
                graph.add_node(Node(str(c.prefix), label=f'{c.output}'))
            graph.add_edge(Edge(str(curr.prefix), str(c.prefix), label=f'{i}'))
            if c.prefix not in visited:
                queue.append(c)
            visited.add(c.prefix)

    graph.add_node(Node('__start0', shape='none', label=''))
    graph.add_edge(Edge('__start0', str(root_node.prefix), label=''))

    graph.write(path=path, format='pdf')