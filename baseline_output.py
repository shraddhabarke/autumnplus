from typing import *
import functools, itertools
from read_trace import read_trace, Transition
from z3 import *

def bits_to_bv(bits):
    int_val = sum(1 << i for i, bit in enumerate(bits) if bit)
    #print("Int-Val:", [1 << i for i, bit in enumerate(bits) if bit])
    return BitVecVal(int_val, len(bits))

def declare(trace, max_lines):
    print("Maxlines:", max_lines)
    s = Solver()
    action_ids = {}
    for t in trace:
        for a in t.possible_actions:
            if a not in action_ids:
                action_ids[a] = len(action_ids)
    
    N, preds = len(trace), len(trace[0].bits) # number of frames # number of predicates
    K = len(action_ids.keys()) # number of all possible actions 

    # line can either correspond to action or state; and can be split on predicates or states. 
    # Children are always one of the possible lines!
    LineType = Datatype('LineType')
    LineType.declare('Action', ('action_val', IntSort()))
    LineType.declare('State', ('state_val', IntSort()))
    LineType.declare('IsState', ('isState_val', IntSort()))
    LineType.declare('Pred', ('pred_val', IntSort()))
    LineType = LineType.create()

    # uninterpreted functions
    line_fn = Function('line_fn', IntSort(), LineType) 
    child_fn = Function('child_fn', LineType, IntSort(), IntSort()) # line_type, child_number -> child_line_number
    line_val = Function('line_val', IntSort(), BitVecSort(len(trace[0].bits)), LineType) #lineVal(line_num, frame) ∈ {possible actions in frame}

    # transition function for states
    transition_fn = Function('T', IntSort(), Preds, IntSort())
    output_fn = Function('O', IntSort(), Preds, IntSort())
    # structural constraints for output decision tree
    for j in range(max_lines):
        line = line_fn(j)
        # if line-type is action, ensure it's within the range of possible action ids
        s.add(Implies(LineType.is_Action(line), And(LineType.action_val(line) >= 0, LineType.action_val(line) < len(action_ids))))
        
        # if line-type is predicate, range should be in valid predicates
        s.add(Implies(LineType.is_Pred(line), And(LineType.pred_val(line) >= 0, LineType.pred_val(line) < preds)))

        all_child_constraints = []
        for child_idx in range(2): # loop over the two children
            child_line = child_fn(line, child_idx)
            if j >= 1:
                s.add(And(0 <= child_line, child_line < j)) # the child_line should always be above the parent
            child_action_constraints = [Implies(LineType.is_Action(line_fn(child_line)), 
                                   And(LineType.action_val(line) >= 0, LineType.action_val(line) < len(action_ids)))] # probably redundant predicates for child lines
            child_pred_constraints = And(LineType.is_Pred(line_fn(child_line)), 0 <= LineType.pred_val(line_fn(child_line)), 
                                             LineType.pred_val(line_fn(child_line)) < preds)
            child_logic = And(*child_action_constraints, child_pred_constraints)
            all_child_constraints.append(child_logic)
            
        s.add(Implies(LineType.is_Pred(line), And(*all_child_constraints))) # will have children only if LineType is Pred or State

    # behavioral constraints for output decision tree
    # TODO: Ignore this function
    def evaluate_line(line_number, frame, depth=0, max_depth=max_lines):
        frame_bv = bits_to_bv(frame.bits)
        if depth < max_depth:
            line = line_fn(line_number)
            is_action_condition = LineType.is_Action(line)
            is_pred_condition = LineType.is_Pred(line)
            pred_val = LineType.pred_val(line)
            # Base Case: Action
            possible_actions = [action_ids[a] for a in frame.possible_actions]
            s.add(Implies(is_action_condition, Or([LineType.action_val(line_val(line_number, frame_bv)) == action for action in possible_actions])))

            true_child_line = child_fn(line, 0) # for true child

            # Implication for true child in case of a predicate
            possible_pred_vals = [i for i in range(preds)]
            for idx in possible_pred_vals:
                s.add(Implies(And(is_pred_condition, pred_val == idx, Extract(idx, idx, frame_bv) == 1), 
                true_child_line == evaluate_line(true_child_line, frame, depth+1, max_depth)))
        
            # Implication for false child in case of a predicate
            false_child_line = child_fn(line, 1) # for false child
            false_child_evaluation = evaluate_line(false_child_line, frame, depth+1, max_depth)
            for idx in possible_pred_vals:
                s.add(Implies(And(is_pred_condition, pred_val == idx, Extract(idx, idx, frame_bv) == 0), 
                  line_val(false_child_line, frame_bv) == false_child_evaluation))
            
    def evaluate_lines(line_number, frame):
        frame_bv = bits_to_bv(frame.bits)
        for current_line_number in range(line_number, 0, -1):
            line = line_fn(current_line_number)
            is_action_condition = LineType.is_Action(line)
            is_pred_condition = LineType.is_Pred(line)
            pred_val = LineType.pred_val(line)

            # Base Case: Action
            possible_actions = [action_ids[a] for a in frame.possible_actions]
            s.add(Implies(is_action_condition, 
                      Or([LineType.action_val(line_val(current_line_number, frame_bv)) == action for action in possible_actions])))

            # Evaluate predicates
            possible_pred_vals = [i for i in range(preds)]
            for idx in possible_pred_vals:
                true_child_line = child_fn(line, 0)  # for true child
                s.add(Implies(And(is_pred_condition, pred_val == idx, Extract(idx, idx, frame_bv) == 1),
                          line_fn(true_child_line) == line_val(true_child_line, frame_bv)))
                print("true!", Implies(And(is_pred_condition, pred_val == idx, Extract(idx, idx, frame_bv) == 1),
                          line_fn(true_child_line) == line_val(true_child_line, frame_bv)))
                
                false_child_line = child_fn(line, 1)  # for false child
                s.add(Implies(And(is_pred_condition, pred_val == idx, Extract(idx, idx, frame_bv) == 0),
                          line_fn(false_child_line) == line_val(false_child_line, frame_bv)))

                print("false!", Implies(And(is_pred_condition, pred_val == idx, Extract(idx, idx, frame_bv) == 0),
                          line_fn(false_child_line) == line_val(false_child_line, frame_bv)))

    for frame in trace:
        evaluate_lines(max_lines - 1, frame)

    # TODO: State lines
    if s.check() == unsat:
        print("unsat!")
        return s, None
    model = s.model()
    return s, model

def solve(trace: list[Transition]):
    for max_lines in itertools.count(4):
        solver, model = declare(trace, max_lines)
        if model is not None or max_lines==20: return model

if __name__ == '__main__':
    trace = read_trace('gravity_i', 5)[2:]
    print("Trace:", trace)
    model = solve(trace)
    print(model)