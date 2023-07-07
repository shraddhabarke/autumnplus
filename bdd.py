from dd.autoref import BDD

traces = [
    ('00', 0),
    ('01', 1),
    ('10', 1),
    ('11', 0)
]
print(traces)
bdd = BDD()
for i in range(len(traces[0][0])):
    bdd.declare(f'x{i}')

top_node = bdd.add_expr('False')

for input, output in traces:
    # Form a boolean expression for this path
    path_exprs = [(f'x{i}' if bit == '1' else f'!x{i}') for i, bit in enumerate(input)]
    path_expr = " & ".join(path_exprs)
    # Construct expression for each trace. If output is 1, use the path expression directly.
    # Otherwise, negate the entire path expression.
    expr = path_expr if output == 1 else f'!({path_expr})'
    node = bdd.add_expr(expr)

bdd.dump('test.png', roots=[node])
