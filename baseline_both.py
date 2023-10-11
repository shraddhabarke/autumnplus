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
    Pred = BitVecSort(len(trace[0].bits))
    K = len(action_ids.keys()) # number of all possible actions 
    states = [Int(f'state_{i}') for i in range(N+1)]
    
    # line can either correspond to action or state; and can be split on predicates or states. 
    # Children are always one of the possible lines!

    LineType = Datatype('LineType')
    LineType.declare('Action', ('action_val', IntSort()))
    LineType.declare('State', ('state_val', IntSort()))
    LineType.declare('StatePred', ('spred_val', IntSort())) # splits on state
    LineType.declare('Pred', ('pred_val', IntSort()))
    LineType = LineType.create()

    # uninterpreted functions
    line_fn = Function('line_fn', IntSort(), LineType) 
    child_fn = Function('child_fn', LineType, IntSort(), IntSort()) # line_type, child_number -> child_line_number
    line_val = Function('line_val', IntSort(), BitVecSort(len(trace[0].bits)), LineType) #lineVal(line_num, frame) ∈ {possible actions in frame}
    state_at = Function('state_at', IntSort(), BitVecSort(len(trace[0].bits)), LineType) #stateAt(line_num, frame) = state[frame]
    
    # State constraints
    transition_fn = Function('T', IntSort(), Pred, IntSort())
    output_fn = Function('O', IntSort(), Pred, IntSort())
    for i in range(len(trace)):
        predicates = functools.reduce(lambda x, y: 2*x + y, trace[i].bits)
        s.add(transition_fn(states[i], predicates) == states[i+1])

    # structural constraints for output decision tree
    for j in range(max_lines):
        line = line_fn(j)
        # if line-type is action, ensure it's within the range of possible action ids
        s.add(Implies(LineType.is_Action(line), And(LineType.action_val(line) >= 0, LineType.action_val(line) < len(action_ids))))
        
        s.add(Implies(LineType.is_State(line), And(LineType.state_val(line) >=0), LineType.state_val(line) < N)) # not a tight constraint, change it to number of unique elements in states?
        # if line-type is predicate, range should be in valid predicates
        s.add(Implies(LineType.is_Pred(line), And(LineType.pred_val(line) >= 0, LineType.pred_val(line) < preds)))

        s.add(Implies(LineType.is_StatePred(line), And(LineType.spred_val(line) >= 0, LineType.spred_val(line) < N)))

        all_child_constraints = []
        for child_idx in range(2): # loop over the two children
            child_line = child_fn(line, child_idx)
            if j >= 1:
                s.add(And(0 <= child_line, child_line < j)) # the child_line should always be above the parent
            child_action_constraints = [Implies(LineType.is_Action(line_fn(child_line)), 
                                   And(LineType.action_val(line) >= 0, LineType.action_val(line) < len(action_ids)))] # probably redundant predicates for child lines
            child_state_constraints = And(LineType.is_State(line_fn(child_line)), 
                                   And(LineType.state_val(line) >= 0, LineType.state_val(line) < N)) # TODO: change
            child_pred_constraints = And(LineType.is_Pred(line_fn(child_line)), 0 <= LineType.pred_val(line_fn(child_line)), 
                                             LineType.pred_val(line_fn(child_line)) < preds)
            child_logic = And(*child_action_constraints, child_state_constraints, child_pred_constraints)
            all_child_constraints.append(child_logic)

        s.add(Implies(LineType.is_Pred(line), And(*all_child_constraints))) # will have children only if LineType is Pred or State

    # behavioral constraints for output decision tree
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

    #TODO: currently children of states can't be preds and vice versa
    def evaluate_state_lines(line_number, i):
        frame_bv = bits_to_bv(trace[i].bits)

        for current_line_number in range(line_number, 0, -1):
            line = line_fn(current_line_number)
            is_state_condition = LineType.is_State(line)
            is_spred_condition = LineType.is_StatePred(line)
            # Base Case: State
            state_value = LineType.state_val(line)
            spred_value = LineType.spred_val(line)
            s.add(Implies(is_state_condition, 
                      state_value == states[i])) # base case, equals the state at the frame

            # Base Case: StatePred
            for state in states:
                true_child_line = child_fn(line, 0)  # for true child
                s.add(Implies(And(is_spred_condition, spred_value == states[i]),
                          line_fn(true_child_line) == line_val(true_child_line, frame_bv))) 
            
                false_child_line = child_fn(line, 1)  # for false child
                s.add(Implies(And(is_spred_condition, spred_value == states[i]),
                          line_fn(false_child_line) == line_val(false_child_line, frame_bv)))  

    for frame in trace:
        evaluate_lines(max_lines - 1, frame)
        evaluate_state_lines(max_lines - 1, i)

    if s.check() == unsat:
        print("unsat!")
        return s, None
    model = s.model()
    return s, model

def solve(trace: list[Transition]):
    for max_lines in itertools.count(4):
        solver, model = declare(trace, max_lines)
        if model is not None: return model

if __name__ == '__main__':
    trace = read_trace('gravity_i', 5)[2:]
    print("Trace:", trace)
    model = solve(trace)
    print(model)