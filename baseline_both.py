from typing import *
import functools, itertools
from read_trace import read_trace, Transition
from z3 import *

def bits_to_bv(bits):
    int_val = sum(1 << i for i, bit in enumerate(bits) if bit)
    #print("Int-Val:", [1 << i for i, bit in enumerate(bits) if bit])
    return BitVecVal(int_val, len(bits))

def declare(trace, max_lines):
    s = Solver()
    action_ids = {}
    for t in trace:
        for a in t.possible_actions:
            if a not in action_ids:
                action_ids[a] = len(action_ids)
    N, preds = len(trace), len(trace[0].bits) # number of frames # number of predicates
    Pred = BitVecSort(len(trace[0].bits))
    K = len(action_ids.keys()) # number of all possible actions 
    states = [Int(f'state_{i}') for i in range(N+1)]
    # line can either correspond to action or state; and can be split on predicates or states. 
    # Children are always one of the possible lines!
    LineType = Datatype('LineType')
    LineType.declare('Action', ('action_val', IntSort()))
    LineType.declare('State', ('state_val', IntSort()))
    LineType.declare('BranchPred', ('branch_pred', IntSort())) # splits on state
    LineType.declare('BranchState', ('branch_state', IntSort()))
    LineType = LineType.create()

    # uninterpreted functions
    line_fn = Function('line_fn', IntSort(), LineType)
    left_child = Function('left_child', IntSort(), IntSort())
    right_child = Function('right_child', IntSort(), IntSort())
    line_val = Function('line_val', IntSort(), Pred, IntSort(), LineType)

    #line_val(line_number, predicate, state, LineType)
    #line_val(line_num, frame) ∈ {possible actions in frame}

    # structural constraints for decision trees
    for j in range(max_lines):
        line = line_fn(j)
        # if line-type is action, ensure it's within the range of possible action ids/predicates
        s.add(Implies(LineType.is_Action(line), And(LineType.action_val(line) >= 0, LineType.action_val(line) < len(action_ids))))
        s.add(Implies(LineType.is_BranchPred(line), And(LineType.branch_pred(line) >= 0, LineType.branch_pred(line) < preds)))
        if j >= 1:
            s.add(Implies(Or(LineType.is_BranchPred(line), LineType.is_BranchState(line)),
            And(And(0 <= left_child(j), left_child(j) < j), And(0 <= right_child(j), right_child(j) < j))))

    for j in range(0, max_lines):
        for i in range(N):
            line = line_fn(j)
            is_action_condition = LineType.is_Action(line)
            is_state_condition = LineType.is_State(line)
            is_branchpred_condition = LineType.is_BranchPred(line)
            is_branchstate_condition = LineType.is_BranchState(line)
            branch_pred = LineType.branch_pred(line)
            branch_state = LineType.branch_state(line)
            frame_bv = bits_to_bv(trace[i].bits)

            s.add(Implies(is_action_condition, line_val(j, frame_bv, states[i]) == line_fn(j)))
            s.add(Implies(is_state_condition, line_val(j, frame_bv, states[i]) == line_fn(j)))
            possible_pred_vals = [i for i in range(preds)]
            for idx in possible_pred_vals: # evaluating all possible predicates
                s.add(Implies(And(is_branchpred_condition, branch_pred == idx, Extract(idx, idx, frame_bv) == 1),
                          line_val(j, frame_bv, states[i]) == line_val(left_child(j), frame_bv, states[i])))
                s.add(Implies(And(is_branchpred_condition, branch_pred == idx, Extract(idx, idx, frame_bv) == 0),
                          line_val(j, frame_bv, states[i]) == line_val(right_child(j), frame_bv, states[i])))

                s.add(If(And(is_branchstate_condition, branch_state == states[i]),
                        line_val(j, frame_bv, states[i]) == line_val(left_child(j), frame_bv, states[i]),
                        line_val(j, frame_bv, states[i]) == line_val(right_child(j), frame_bv, states[i])))

    # evaluate correctly
    for i in range(N):
        frame_bv = bits_to_bv(trace[i].bits)
        possible_actions = [action_ids[a] for a in trace[i].possible_actions]
        # Top Action - action_val could belong to one of the possible actions
        possible_actions = [action_ids[a] for a in trace[i].possible_actions]
        s.add(And(LineType.is_Action(line_val(max_lines - 2, frame_bv, states[i])),
        Or([LineType.action_val(line_val(max_lines - 2, frame_bv, states[i])) ==
            action for action in possible_actions])))

        # Top State
        s.add(And(LineType.is_State(line_val(max_lines - 1, frame_bv, states[i])),
            LineType.state_val(line_val(max_lines - 1, frame_bv, states[i])) == states[i+1]))

    if s.check() == unsat:
        print("unsat!")
        return s, None
    model = s.model()
    return s, model

def solve(trace: list[Transition]):
    for max_lines in itertools.count(2):
        if max_lines > 20: break
        solver, model = declare(trace, max_lines)
        if model is not None: return model

if __name__ == '__main__':
    trace = read_trace('test_', 1)#[1:]
    print("Trace:", trace)
    model = solve(trace)
    print(model)