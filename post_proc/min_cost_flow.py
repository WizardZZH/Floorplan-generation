from ortools.linear_solver import pywraplp

import numpy as np
from ortools.graph.python import min_cost_flow

def MIP_form():
    # Create the mip solver with the SCIP backend.
    solver = pywraplp.Solver.CreateSolver('SCIP')
    if not solver:
        return

    infinity = solver.infinity()
    # x and y are integer non-negative variables.
    a_f1 = solver.IntVar(0, 2, 'a_f1')
    # f1_a = solver.IntVar(0.0, 0, 'f1_a')

    b_f1 = solver.IntVar(0, 2, 'b_f1')
    # f1_b = solver.IntVar(0.0, 0, 'f1_b')
    
    c_f1 = solver.IntVar(0, 2, 'c_f1')
    # f1_b = solver.IntVar(0.0, 0, 'f1_b')


    # a_f1 = solver.IntVar(0.0, 0, 'a_f1')
    f0_a = solver.IntVar(0, 2, 'f0_a')

    # b_f1 = solver.IntVar(0.0, 0, 'b_f1')
    f0_b = solver.IntVar(0, 2, 'f0_b')

    # c_f1 = solver.IntVar(0.0, 0, 'c_f1')
    f0_c = solver.IntVar(0, 2, 'f0_c')


    e_ac = solver.IntVar(0.0, infinity, 'e_ac')
    e_ca = solver.IntVar(0.0, infinity, 'e_ca')

    e_ab = solver.IntVar(0.0, infinity, 'e_ab')
    e_ba = solver.IntVar(0.0, infinity, 'e_ba')

    e_bc = solver.IntVar(0.0, infinity, 'e_bc')
    e_cb = solver.IntVar(0.0, infinity, 'e_cb')




    print('Number of variables =', solver.NumVariables())

    # vex = 4
    solver.Add(3 + 1 + a_f1 - f0_a == 4)
    solver.Add(3 + 1 + b_f1 - f0_b == 4)
    solver.Add(3 + 1 + c_f1 - f0_c == 4)

    # face
    solver.Add(-1 - a_f1 - 1 - b_f1 - 1 - c_f1 - e_ca - e_ab - e_bc + e_ac + e_ba + e_cb == -2 )

    solver.Add(-3 - f0_a - 3 - f0_b - 3 - f0_c + e_ca  + e_ab + e_bc - e_ac - e_ba - e_cb == -10 )


    # # vex = 4
    # solver.Add( a_f1 + f0_a == 4)
    # solver.Add( b_f1 + f0_b == 4)
    # solver.Add( c_f1 + f0_c == 4)

    # # face
    # solver.Add( - a_f1 - b_f1  - c_f1 - e_ca - e_ab - e_bc + e_ac + e_ba + e_cb == -2 )

    # solver.Add( - f0_a  - f0_b - f0_c + e_ca  + e_ab + e_bc - e_ac - e_ba - e_cb == -10 )

    print('Number of constraints =', solver.NumConstraints())

    # Maximize x + 10 * y.
    solver.Maximize(-1.0*(a_f1 + b_f1 + c_f1 + f0_a + f0_b + f0_c +e_ac + e_ca + e_ab + e_ba + e_bc + e_cb))
    #solver.Maximize(-1.0*( e_ac + e_ca + e_ab + e_ba + e_bc + e_cb))

    status = solver.Solve()

    if status == pywraplp.Solver.OPTIMAL:
        print('Solution:')
        print('Objective value =', solver.Objective().Value())
        print('a_f1 =', a_f1.solution_value())
        print('f0_a =', f0_a.solution_value())
        
        print('b_f1 =', b_f1.solution_value())
        print('f0_b =', f0_b.solution_value())

        print('c_f1 =', c_f1.solution_value())
        print('f0_c =', f0_c.solution_value())

        print('e_ac =', e_ac.solution_value())
        print('e_ca =', e_ca.solution_value())

        print('e_ab =', e_ab.solution_value())
        print('e_ba =', e_ba.solution_value())

        print('e_bc =', e_bc.solution_value())
        print('e_cb =', e_cb.solution_value())

    else:
        print('The problem does not have an optimal solution.')

    # print('\nAdvanced usage:')
    # print('Problem solved in %f milliseconds' % solver.wall_time())
    # print('Problem solved in %d iterations' % solver.iterations())
    # print('Problem solved in %d branch-and-bound nodes' % solver.nodes())


def min_cost_flow_form(network):
    """MinCostFlow simple interface example."""
    # Instantiate a SimpleMinCostFlow solver.
    smcf = min_cost_flow.SimpleMinCostFlow()

    # Define four parallel arrays: sources, destinations, capacities,
    # and unit costs between each pair. For instance, the arc from node 0
    # to node 1 has a capacity of 15.
    # start_nodes = np.array([0, 0, 1, 1, 2, 2, 3, 3, 3, 4, 4, 4, 3, 3, 3, 4, 4, 4])
    # end_nodes =   np.array([3, 4, 3, 4, 3, 4, 4, 4 ,4, 3, 3, 3, 0, 1, 2, 0, 1 ,2])
    # capacities =  np.array([2, 0 ,2 ,0, 2, 0, 10, 10, 10, 10, 10, 10, 0,0,0,2,2,2])
    # unit_costs = np.ones(18)

    # # Define an array of supplies at each node.
    # supplies = [0, 0, 0, 1 , -1]

    start_nodes = network['start_nodes']
    end_nodes =   network['end_nodes']
    capacities = network['capacities']
    unit_costs = network['unit_costs']
    # Define an array of supplies at each node.
    supplies = network['supplies']

    # Add arcs, capacities and costs in bulk using numpy.
    all_arcs = smcf.add_arcs_with_capacity_and_unit_cost(
        start_nodes, end_nodes, capacities, unit_costs)

    # Add supply for each nodes.
    smcf.set_nodes_supply(np.arange(0, len(supplies)), supplies)

    # Find the min cost flow.
    status = smcf.solve()

    if status != smcf.OPTIMAL:
        print('There was an issue with the min cost flow input.')
        print(f'Status: {status}')
        return 1,1
    #print(f'Minimum cost: {smcf.optimal_cost()}')
    #print('')
    #print(' Arc    Flow / Capacity Cost')
    solution_flows = smcf.flows(all_arcs)
    costs = solution_flows * unit_costs
    # for arc, flow, cost in zip(all_arcs, solution_flows, costs):
    #     print(
    #         f'{smcf.tail(arc):1} -> {smcf.head(arc)}  {flow:3}  / {smcf.capacity(arc):3}       {cost}'
    #     )

    return solution_flows,0

if __name__ == '__main__':
    main()