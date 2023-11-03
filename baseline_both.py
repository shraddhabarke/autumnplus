from typing import *
from read_trace import read_trace, Transition
from z3 import *
import time

LINE_LIMIT = 20

def bits_to_bv(bits):
    int_val = sum(1 << i for i, bit in enumerate(bits) if bit)
    #print("Int-Val:", [1 << i for i, bit in enumerate(bits) if bit])
    return BitVecVal(int_val, len(bits))

# A class to store the solver and type declarations:
class SolverState:
    def init(self, trace):
        self.solver = Solver()
        self.solver.set(unsat_core=True)
        self.action_ids = {}
        for t in trace:
            for a in t.possible_actions:
                if a not in self.action_ids:
                    self.action_ids[a] = len(self.action_ids)

        self.N, self.preds = len(trace), len(trace[0].bits) # number of frames # number of predicates
        self.Pred = BitVecSort(len(trace[0].bits))
        self.K = len(self.action_ids.keys()) # number of all possible actions 
        self.states = [Int(f'state_{i}') for i in range(self.N+1)]        

        # line can either correspond to action or state; and can be split on predicates or states. 
        # Children are always one of the possible lines!
        self.LineType = Datatype('LineType')
        self.LineType.declare('Action', ('action_val', IntSort()))
        self.LineType.declare('State', ('state_val', IntSort()))
        self.LineType.declare('BranchPred', ('branch_pred', IntSort())) # splits on state
        self.LineType.declare('BranchState', ('branch_state', IntSort()))
        self.LineType = self.LineType.create()

        # uninterpreted functions
        self.line_fn = Function('line_fn', IntSort(), self.LineType)
        self.left_child = Function('left_child', IntSort(), IntSort())
        self.right_child = Function('right_child', IntSort(), IntSort())
        self.line_val = Function('line_val', IntSort(), self.Pred, IntSort(), self.LineType)

def add_constraint_with_label(solver, constraint, label):
    solver.assert_and_track(constraint, label)

def gen_constraints(ss: SolverState, trace: list[Transition], max_lines: int):
    '''Add constraints to the solver in ss that would produce the trace using a program with max_lines lines.'''

    s, LineType, left_child, right_child, line_fn, line_val, states, preds, action_ids, N, K = \
        ss.solver, ss.LineType, ss.left_child, ss.right_child, ss.line_fn, ss.line_val, ss.states, ss.preds, ss.action_ids, ss.N, ss.K
    
    # Structural constraints on states
    for i in range(N+1):
        # States are non-negative
        s.add(states[i] >= 0)
        add_constraint_with_label(s, ss.states[i] >= 0, f'state_{i}_non_negative')
        # If current state is n > 0, then one of the previous states is n-1 (optional; makes states consecutive)
        # s.add(Implies(states[i] > 0, Or([states[j] == states[i] - 1 for j in range(i)])))
    # the very first state is zero (without loss of generality); optional
    s.add(states[0] == 0)
    add_constraint_with_label(s, ss.states[0] == 0, 'first_state_zero')

    # structural constraints for decision trees
    for j in range(max_lines):
        line = line_fn(j)
        # if line-type is action, ensure it's within the range of possible action ids/predicates
        s.add(Implies(LineType.is_Action(line), And(LineType.action_val(line) >= 0, LineType.action_val(line) < K)))
        s.add(Implies(LineType.is_BranchPred(line), And(LineType.branch_pred(line) >= 0, LineType.branch_pred(line) < preds)))
        s.add(Implies(Or(LineType.is_BranchPred(line), LineType.is_BranchState(line)),
            And(And(0 <= left_child(j), left_child(j) < j), And(0 <= right_child(j), right_child(j) < j))))

        add_constraint_with_label(s, Implies(LineType.is_Action(line), And(LineType.action_val(line) >= 0, LineType.action_val(line) < K)), f'struct_action_line_{j}')
        add_constraint_with_label(s, Implies(LineType.is_BranchPred(line), And(LineType.branch_pred(line) >= 0, LineType.branch_pred(line) < preds)),
                                f'struct_branchpred_line_{j}')
        add_constraint_with_label(s, Implies(Or(LineType.is_BranchPred(line), LineType.is_BranchState(line)),
                                             And(And(0 <= left_child(j), left_child(j) < j),
                                                 And(0 <= right_child(j), right_child(j) < j))),
                                  f'struct_children_line_{j}')

    # semantics of the decision tree
    for j in range(0, max_lines):
        for i in range(N):
            line = line_fn(j)            
            frame_bv = bits_to_bv(trace[i].bits)

            # If the line is an action, its value it itself
            s.add(Implies(LineType.is_Action(line), line_val(j, frame_bv, states[i]) == line_fn(j)))
            add_constraint_with_label(s, Implies(LineType.is_Action(line), line_val(j, frame_bv, states[i]) == line_fn(j)), f'action_line_{j}_frame_{i}')
            # If the line is a state assignment, its value is itself, unless it's negative (interpreted as stay)
            s.add(Implies(LineType.is_State(line),
                line_val(j, frame_bv, states[i]) == If(LineType.state_val(line_fn(j)) < 0, LineType.State(states[i]), line_fn(j))))
            add_constraint_with_label(s, Implies(LineType.is_State(line),
            line_val(j, frame_bv, states[i]) == If(LineType.state_val(line) < 0, LineType.State(states[i]), line_fn(j))), f'state_line_{j}_frame_{i}')

            for idx in range(preds): # evaluating all possible predicates
                if trace[i].bits[idx]:
                    s.add(Implies(And(LineType.is_BranchPred(line), LineType.branch_pred(line) == idx),
                          line_val(j, frame_bv, states[i]) == line_val(left_child(j), frame_bv, states[i])))

                    add_constraint_with_label(s, Implies(And(LineType.is_BranchPred(line), LineType.branch_pred(line) == idx),
                        line_val(j, frame_bv, states[i]) == line_val(left_child(j), frame_bv, states[i])),
                        f'branch_true_line_{j}_pred_{idx}_frame_{i}')
                else:
                    s.add(Implies(And(LineType.is_BranchPred(line), LineType.branch_pred(line) == idx),
                          line_val(j, frame_bv, states[i]) == line_val(right_child(j), frame_bv, states[i])))
                    add_constraint_with_label(s, Implies(And(LineType.is_BranchPred(line), LineType.branch_pred(line) == idx),
                        line_val(j, frame_bv, states[i]) == line_val(right_child(j), frame_bv, states[i])),
                        f'branch_false_line_{j}_pred_{idx}_frame_{i}')
                
            s.add(Implies(LineType.is_BranchState(line),
                          line_val(j, frame_bv, states[i]) == 
                              If(states[i] == LineType.branch_state(line), 
                                 line_val(left_child(j), frame_bv, states[i]), 
                                 line_val(right_child(j), frame_bv, states[i]))))

            add_constraint_with_label(s, Implies(LineType.is_BranchState(line),
                            line_val(j, frame_bv, states[i]) == If(states[i] == LineType.branch_state(line),
                            line_val(left_child(j), frame_bv, states[i]),
                            line_val(right_child(j), frame_bv, states[i]))),
                            f'branchstate_line_{j}_frame_{i}')

    # evaluate correctly
    action_root = max_lines - 1
    state_root = max_lines - 2
    for i in range(N):
        frame_bv = bits_to_bv(trace[i].bits)
        possible_actions = [action_ids[a] for a in trace[i].possible_actions]
        # Top Action is one of the possible actions
        s.add(And(LineType.is_Action(line_val(action_root, frame_bv, states[i])),
              Or([LineType.action_val(line_val(action_root, frame_bv, states[i])) == action for action in possible_actions])))
        add_constraint_with_label(s, And(LineType.is_Action(line_val(action_root, frame_bv, states[i])),
              Or([LineType.action_val(line_val(action_root, frame_bv, states[i])) == action for action in possible_actions])),
              f'top_action_frame_{i}')

        # Top State is the current state
        s.add(And(LineType.is_State(line_val(state_root, frame_bv, states[i])),
            LineType.state_val(line_val(state_root, frame_bv, states[i])) == states[i+1]))
        add_constraint_with_label(s, And(LineType.is_State(line_val(state_root, frame_bv, states[i])),
            LineType.state_val(line_val(state_root, frame_bv, states[i])) == states[i+1]), f'top_state_frame_{i}')
        # If no events happened, then the state should stay the same
        if frame_bv == 0:
            s.add(states[i] == states[i+1])
            add_constraint_with_label(s, states[i] == states[i+1], f'stay{i}')

    # Symmetry breaking constraints:

    # Line #0 is always "stay" (optional)
    s.add(line_fn(0) == LineType.State(-1))
    # The next K lines are all different actions:
    for j in range(K):
        s.add(line_fn(j+1) == LineType.Action(j))

def gen_state_constraints(ss: SolverState, num_states: int):
    '''Add constraints that are specific to the number of states in the automaton.'''
    # The next num_states lines after the action lines are all different states assignments:
    for j in range(num_states):
        ss.solver.add(ss.line_fn(ss.K+1+j) == ss.LineType.State(j))

    for i in range(ss.N+1):
        ss.solver.add(ss.states[i] < num_states)

def solve(trace: list[Transition]):
    ss = SolverState()
    ss.init(trace)
    # Iterate over the size of the decisions tree
    # (we start from 2 because we need at least one action and one state assignment)
    for num_lines in range(2, LINE_LIMIT):
        print("Trying max_lines =", num_lines)
        ss.solver.push() # save the current state of the solver
        gen_constraints(ss, trace, num_lines)

        # Iterate over the number of states in the automaton;
        # (this makes things faster because we can fix all the state assignment lines)
        max_states = num_lines - ss.K # We can't have more states than lines to put their assignments in
        for num_states in range(1, max_states):
            print("\tTrying num_states =", num_states)
            ss.solver.push()
            gen_state_constraints(ss, num_states)

            if ss.solver.check() == unsat:
                print("\tUNSAT")
                print(ss.solver.unsat_core())
            else:
                return ss.solver.model()
            ss.solver.pop() # pop num-state-specific constraints
        ss.solver.pop() # pop size-specific constraints

if __name__ == '__main__':
    trace = read_trace('test1_', 1)
    print("Trace:", trace)
    model = solve(trace)
    print("Printing Model:")
    print(model)