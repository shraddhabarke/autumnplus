import time
from bisect import insort
from typing import Union

from Automaton import DeterministicAutomaton
from rpni_helper_functions import to_automaton, createPTA

class RPNI:
    def __init__(self, data, automaton_type, print_info=True):
        self.data = data
        self.automaton_type = automaton_type
        self.print_info = print_info

        pta_construction_start = time.time()
        self.root_node = createPTA(data, automaton_type)
        if self.print_info:
            print(f'PTA Construction Time: {round(time.time() - pta_construction_start, 2)}')

    def run_rpni(self):
        start_time = time.time()

        red = [self.root_node]                # initialize the red states with the root node
        blue = list(red[0].children.values()) # all children of the root node as initial blue states
        while blue:
            lex_min_blue = min(list(blue))  # TODO: could be a point where we attempt to merge the cheapest blue state?
            merged = False
            for red_state in red:
                if not self._compatible_states(red_state, lex_min_blue):
                    continue
                self._merge(red_state, lex_min_blue)
                merged = True
                break

            if not merged:
                insort(red, lex_min_blue)     # if it's not merged, the blue state becomes a red (fixed) state
                if self.print_info:
                    print(f'\rCurrent automaton size: {len(red)}\n', end="")

            blue.clear() # clear the blue states

            for r in red:
                for c in r.children.values():
                    if c not in red:
                        blue.append(c)        # add all children of the red states to the blue states

        if self.print_info:
            print(f'\nRPNI Learning Time: {round(time.time() - start_time, 2)}')
            print(f'RPNI Learned {len(red)} state automaton.')

        assert sorted(red, key=lambda x: len(x.prefix)) == red
        return to_automaton(red, self.automaton_type)

    def _compatible_states(self, red_node, blue_node):
        #Only allow merging of states that have same output(s)
        red_io = red_node.output
        blue_io = blue_node.output

        for common_i in set(red_io.keys()).intersection(blue_io.keys()):
            if red_io[common_i] != blue_io[common_i]:
                return False
        return True

    def _merge(self, red_node, lex_min_blue, copy_nodes=False):
        #Merge two states and return the root node of resulting model
        root_node = self.root_node.copy() if copy_nodes else self.root_node
        lex_min_blue = lex_min_blue.copy() if copy_nodes else lex_min_blue

        red_node_in_tree = root_node

        for p in red_node.prefix:
            red_node_in_tree = red_node_in_tree.children[p] #traverse the tree along the path defined by the red node

        to_update = root_node

        for p in lex_min_blue.prefix[:-1]:
            to_update = to_update.children[p] # traverse the tree till the parent of the blue node

        to_update.children[lex_min_blue.prefix[-1]] = red_node_in_tree # assign the blue node to the red node, merged!

        self._fold_mealy(red_node_in_tree, lex_min_blue)
        return root_node

    def _fold_mealy(self, red_node, blue_node):
        blue_io_map = blue_node.output

        for io, val in blue_node.children.items():
            if io not in red_node.children.keys():
                red_node.output[io] = blue_io_map[io] # update red node's output for nodes that are children of blue node but not red node

        for io in blue_node.children.keys():
            if io in red_node.children.keys():
                # there's a non-deterministic transition function, so merge further
                self._fold_mealy(red_node.children[io], blue_node.children[io])
            else:
                red_node.children[io] = blue_node.children[io]  # add the remaining children of blue node to red node

def run_RPNI(data, automaton_type, algorithm='classic',
             input_completeness=None, print_info=True) -> Union[DeterministicAutomaton, None]:
    """
    Run RPNI, a deterministic passive model learning algorithm.

    Args:
        data: sequence of input sequences and corresponding label. Eg. [[(i1,i2,i3, ...), label], ...]
        automaton_type: either 'dfa', 'mealy', 'moore'. Note that for 'mealy' machine learning, data has to be prefix-closed.
        algorithm: either 'gsm' (generalized state merging) or 'classic' for base RPNI implementation. GSM is much faster and less resource intensive.
        input_completeness: either None, 'sink_state', or 'self_loop'. If None, learned model could be input incomplete,
        sink_state will lead all undefined inputs form some state to the sink state, whereas self_loop will simply create
        a self loop. In case of Mealy learning output of the added transition will be 'epsilon'.
    Returns:

        Model conforming to the data, or None if data is non-deterministic.
    """
    assert algorithm in {'classic'}
    assert input_completeness in {None, 'self_loop', 'sink_state'}

    rpni = RPNI(data, automaton_type, print_info)

    if rpni.root_node is None:
        print('Data provided to RPNI is not deterministic. Ensure that the data is deterministic, '
                'or consider using Alergia.')
        return None

    learned_model = rpni.run_rpni()

    if not learned_model.is_input_complete():
        if not input_completeness:
            if print_info:
                print('Warning: Learned Model is not input complete (inputs not defined for all states). '
                      'Consider calling .make_input_complete()')
        else:
            if print_info:
                print(f'Learned model was not input complete. Adapting it with {input_completeness} transitions.')
            learned_model.make_input_complete(input_completeness)

    return learned_model