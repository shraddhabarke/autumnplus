from typing import *
import functools, itertools
from read_trace import read_trace, Transition
from z3 import *

def bits_to_bv(bits):
    int_val = sum(1 << i for i, bit in enumerate(bits) if bit)
    #print("Int-Val:", [1 << i for i, bit in enumerate(bits) if bit])
    return BitVecVal(int_val, len(bits))

# A class to store the solver and type declarations:
class SolverState:
    def init(self, trace):
        self.solver = Solver()
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


def gen_constraints(ss: SolverState, trace: list[Transition], max_lines: int):
    '''Add constraints to the solver in ss that would produce the trace using a program with max_lines lines.'''

    s, LineType, left_child, right_child, line_fn, line_val, states, preds, action_ids, N, K = \
        ss.solver, ss.LineType, ss.left_child, ss.right_child, ss.line_fn, ss.line_val, ss.states, ss.preds, ss.action_ids, ss.N, ss.K
    
    # Structural constraints on states
    for i in range(N+1):
        # States are non-negative
        s.add(states[i] >= 0)
        # If current state is n > 0, then one of the previous states is n-1 (optional; makes states consecutive)
        # s.add(Implies(states[i] > 0, Or([states[j] == states[i] - 1 for j in range(i)])))
    # the very first state is zero (without loss of generality); optional
    s.add(states[0] == 0)

    # structural constraints for decision trees
    for j in range(max_lines):
        line = line_fn(j)
        # if line-type is action, ensure it's within the range of possible action ids/predicates
        s.add(Implies(LineType.is_Action(line), And(LineType.action_val(line) >= 0, LineType.action_val(line) < K)))
        s.add(Implies(LineType.is_BranchPred(line), And(LineType.branch_pred(line) >= 0, LineType.branch_pred(line) < preds)))
        s.add(Implies(Or(LineType.is_BranchPred(line), LineType.is_BranchState(line)),
            And(And(0 <= left_child(j), left_child(j) < j), And(0 <= right_child(j), right_child(j) < j))))

    # semantics of the decision tree
    for j in range(0, max_lines):
        for i in range(N):
            line = line_fn(j)            
            frame_bv = bits_to_bv(trace[i].bits)

            # If the line is an action, its value it itself
            s.add(Implies(LineType.is_Action(line), line_val(j, frame_bv, states[i]) == line_fn(j)))

            # If the line is a state assignment, its value is itself, unless it's negative (interpreted as stay)
            s.add(Implies(LineType.is_State(line), 
                line_val(j, frame_bv, states[i]) == If(LineType.state_val(line_fn(j)) < 0, LineType.State(states[i]), line_fn(j))))

            for idx in range(preds): # evaluating all possible predicates
                if trace[i].bits[idx]:
                    s.add(Implies(And(LineType.is_BranchPred(line), LineType.branch_pred(line) == idx),
                          line_val(j, frame_bv, states[i]) == line_val(left_child(j), frame_bv, states[i])))
                else:
                    s.add(Implies(And(LineType.is_BranchPred(line), LineType.branch_pred(line) == idx),
                          line_val(j, frame_bv, states[i]) == line_val(right_child(j), frame_bv, states[i])))
                
            s.add(Implies(LineType.is_BranchState(line),
                          line_val(j, frame_bv, states[i]) == 
                              If(states[i] == LineType.branch_state(line), 
                                 line_val(left_child(j), frame_bv, states[i]), 
                                 line_val(right_child(j), frame_bv, states[i]))))

    # evaluate correctly
    action_root = max_lines - 1
    state_root = max_lines - 2
    for i in range(N):
        frame_bv = bits_to_bv(trace[i].bits)
        possible_actions = [action_ids[a] for a in trace[i].possible_actions]
        # Top Action is one of the possible actions
        s.add(And(LineType.is_Action(line_val(action_root, frame_bv, states[i])),
              Or([LineType.action_val(line_val(action_root, frame_bv, states[i])) == action for action in possible_actions])))

        # Top State is the current state
        s.add(And(LineType.is_State(line_val(state_root, frame_bv, states[i])),
            LineType.state_val(line_val(state_root, frame_bv, states[i])) == states[i+1]))
        
        # If no events happened, then the state should stay the same
        if frame_bv == 0:
            s.add(states[i] == states[i+1])


    # Line #0 is always "stay" (optional)
    s.add(line_fn(0) == LineType.State(-1))
    


def solve(trace: list[Transition]):
    ss = SolverState()
    ss.init(trace)

    for max_lines in itertools.count(2):
        if max_lines > 8: break

        print("Trying max_lines =", max_lines)
        ss.solver.push() # save the current state of the solver
        gen_constraints(ss, trace, max_lines)
        # print(ss.solver)
        if ss.solver.check() == unsat:
            print("UNSAT")
            ss.solver.pop()
        else:
            return ss.solver.model()

if __name__ == '__main__':
    trace = read_trace('test_', 1)#[1:]
    print("Trace:", trace)
    model = solve(trace)
    print(model)