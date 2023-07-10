def rpni_mealy_example():
    import random
    from RPNI import run_RPNI
    #data = [(('a',), 'd'), (('a', 'a',), 'd'), (('a', 'a', 'b'), 'u'), (('a', 'a', 'b', 'a'), 'u'),  (('a', 'a', 'b', 'a', 'a'), 'u'), (('a', 'a', 'b', 'a', 'a', 'c'), 'd'), (('a', 'a', 'b', 'a', 'a', 'c', 'a'), 'd')]
    #data = [(('a',), 'd'), (('a', 'a',), 'd'), (('a', 'a', 'b'), 'u'), (('a', 'a', 'b', 'a'), 'u')]
    #data = [(('epsilon',), 'd'), (('epsilon', 'a',), 'd'), (('epsilon', 'a', 'a',), 'd'), (('epsilon', 'a', 'a', 'b'), 'u'), (('epsilon', 'a', 'a', 'b', 'a'), 'u'),  (('epsilon', 'a', 'a', 'b', 'a', 'a'), 'u'), (('epsilon', 'a', 'a', 'b', 'a', 'a', 'c'), 'd'), (('epsilon', 'a', 'a', 'b', 'a', 'a', 'c', 'a'), 'd')]
    #data = [(('epsilon1',), 'o'), (('epsilon1', 'a1',), 'd1'), (('epsilon1', 'a1', 'a1',), 'd1'), (('epsilon1', 'a1', 'a1', 'b1',), 'u1')]
    data = [(('a',), 'd'), (('a', 'a',), 'd'), (('a', 'a', 'b'), 'u'), (('a', 'a', 'b', 'a'), 'u'),  (('a', 'a', 'b', 'a', 'a'), 'u'), (('a', 'a', 'b', 'a', 'a', 'c'), 'd'), (('a', 'a', 'b', 'a', 'a', 'c', 'a'), 'd')]    

    data1 = [(('moveDown',), ''), 
    (('moveDown', 'moveLeft', 'moveLeft', 'moveLeft', 'moveLeft', 'moveLeft', 'moveRight', 'moveRight', 'moveRight', 'moveUp', 'moveUp', 'moveDown', 'moveDown', 'moveDown', 'moveDown'), '')]    

    rpni_model = run_RPNI(data, automaton_type='mealy', print_info=True)
    rpni_model.save()
    return rpni_model

rpni_mealy_example()