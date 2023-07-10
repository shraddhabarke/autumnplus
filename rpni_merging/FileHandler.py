import os
import sys
import traceback
from pathlib import Path

from pydot import Dot, Node, Edge, graph_from_dot_file

from MealyMachine import MealyMachine

file_types = ['dot', 'png', 'svg', 'pdf', 'string']
automaton_types = {MealyMachine: 'mealy'}

def _wrap_label(label):
    """
    Adds a " " around a label if not already present on both ends.
    """
    if label[0] == '\"' and label[-1] == '\"':
        return label
    return f'\"{label}\"'

def _get_node(state, automaton_type):
    if automaton_type == 'mealy':
        return Node(state.state_id, label=_wrap_label(state.state_id))

def _add_transition_to_graph(graph, state, automaton_type, display_same_state_trans, round_floats):
    if automaton_type == 'mealy':
        for i in state.transitions.keys():
            new_state = state.transitions[i]
            if not display_same_state_trans and new_state.state_id == state.state_id:
                continue
            graph.add_edge(Edge(state.state_id, new_state.state_id, label=_wrap_label(f'{i}/{state.output_fun[i]}')))

def visualize_automaton(automaton, path="LearnedModel", file_type="pdf", display_same_state_trans=True):
    """
    Create a graphical representation of the automaton.
    Function is round in the separate thread in the background.
    If possible, it will be opened by systems default program.

    Args:

        automaton: automaton to be visualized

        path: pathlike or str, file in which visualization will be saved (Default value = "LearnedModel.pdf")

        file_type: type of file/visualization. Can be ['png', 'svg', 'pdf'] (Default value = "pdf")

        display_same_state_trans: if True, same state transitions will be displayed (Default value = True)

    """
    print('Visualization started in the background thread.')

    if len(automaton.states) >= 25:
        print(f'Visualizing {len(automaton.states)} state automaton could take some time.')

    import threading
    visualization_thread = threading.Thread(target=save_automaton_to_file, name="Visualization",
                                            args=(automaton, path, file_type, display_same_state_trans, True, 2))
    visualization_thread.start()


def save_automaton_to_file(automaton, path="LearnedModel", file_type="dot",
                           display_same_state_trans=True, visualize=False, round_floats=None):
    """
    Args:
        automaton: automaton to be saved to file
        path: pathlike or str, file in which visualization will be saved (Default value = "LearnedModel")

        file_type: type of file/visualization. Can be ['dot', 'png', 'svg', 'pdf'] (Default value = "dot)

        display_same_state_trans: True, should not be set to false except from the visualization method
            (Default value = True)

        visualize: visualize the automaton
    """
    path = Path(path)
    file_type = file_type.lower()
    assert file_type in file_types, f"Filetype {file_type} not in allowed filetypes"
    path = path.with_suffix(f".{file_type}")
    if file_type == 'dot' and not display_same_state_trans:
        print("When saving to file all transitions will be saved")
        display_same_state_trans = True
    automaton_type = automaton_types[automaton.__class__]

    graph = Dot(path.stem, graph_type='digraph')
    for state in automaton.states:
        graph.add_node(_get_node(state, automaton_type))

    for state in automaton.states:
        _add_transition_to_graph(graph, state, automaton_type, display_same_state_trans, round_floats)

    # add initial node
    graph.add_node(Node('__start0', shape='none', label=''))
    graph.add_edge(Edge('__start0', automaton.initial_state.state_id, label=''))

    if file_type == 'string':
        return graph.to_string()
    else:
        try:
            graph.write(path=path, format=file_type if file_type != 'dot' else 'raw')
            print(f'Model saved to {path.as_posix()}.')

            if visualize and file_type in {'pdf', 'png', 'svg'}:
                try:
                    import webbrowser
                    webbrowser.open(path.resolve().as_uri())
                except OSError:
                    traceback.print_exc()
                    print(f'Could not open the file {path.as_posix()}.', file=sys.stderr)
        except OSError:
            traceback.print_exc()
            print(f'Could not write to the file {path.as_posix()}.', file=sys.stderr)