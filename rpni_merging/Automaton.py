import copy
import warnings
from abc import ABC
from collections import defaultdict
from typing import Union, TypeVar, Generic, List

class AutomatonState(ABC):

    def __init__(self, state_id):
        """
        Single state of an automaton. Each state consists of a state id, a dictionary of transitions, where the keys are
        inputs and the values are the corresponding target states, and a prefix that leads to the state from the initial
        state.

        Args:

            state_id(Any): used for graphical representation of the state. A good practice is to keep it unique.

        """
        self.state_id = state_id
        self.transitions = None
        self.prefix = None

AutomatonStateType = TypeVar("AutomatonStateType", bound=AutomatonState)

OutputType = TypeVar("OutputType")
InputType = TypeVar("InputType")

class Automaton(ABC, Generic[AutomatonStateType]):
    """
    Abstract class representing an automaton.
    """

    def __init__(self, initial_state: AutomatonStateType, states: List[AutomatonStateType]):
        """
        Args:

            initial_state (AutomatonState): initial state of the automaton
            states (list) : list containing all states of the automaton

        """
        self.initial_state: AutomatonStateType = initial_state
        self.states: List[AutomatonStateType] = states
        self.characterization_set: list = []
        self.current_state: AutomatonStateType = initial_state

    @property
    def size(self):
        return len(self.states)

    def is_input_complete(self) -> bool:
        """
        Check whether all states have defined transition for all inputs
        :return: true if automaton is input complete

        Returns:

            True if input complete, False otherwise

        """
        alphabet = set(self.get_input_alphabet())
        for state in self.states:
            if set(state.transitions.keys()) != alphabet:
                return False
        return True

    def get_input_alphabet(self) -> list:
        """
        Returns the input alphabet
        """
        alphabet = list()
        for s in self.states:
            for i in s.transitions.keys():
                if i not in alphabet:
                    alphabet.append(i)
        return list(alphabet)

    def execute_sequence(self, origin_state, seq):
        self.current_state = origin_state
        return [self.step(s) for s in seq]

    def save(self, file_path='LearnedModel'):
        from FileHandler import save_automaton_to_file
        save_automaton_to_file(self, path=file_path)

    def visualize(self, path='LearnedModel', file_type='pdf', display_same_state_transitions=True):
        from FileHandler import visualize_automaton
        visualize_automaton(self, path, file_type, display_same_state_transitions)

class DeterministicAutomaton(Automaton[AutomatonStateType]):
    pass