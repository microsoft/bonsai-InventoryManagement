'''
This class will be used as a solver 
'''

'''
___author__ = Hossein Khadivi Heris 
Note: Only supported for fixed lead time
'''




from typing import Type
from distutils.command.config import config
import time as t
from mip import Model, xsum, minimize, BINARY, INTEGER, CONTINUOUS
from itertools import product
from sys import stdout as out
from tkinter.tix import INTEGER
from tabulate import tabulate
import numpy as np
from .multi_sku import SupplyChainTopology
from .multi_sku import SKUInfoFactory
from .utils import calc_runtime
class MipSolver:
    def __init__(self, config):
        pass

    @calc_runtime
    def get_mip_action_multiple_sku(self, skus: SKUInfoFactory, ratio: float = 0.1, cost_function: str = "Hybrid", constraint_coupling: bool = 1):
        '''
        Input:SKU object containing generic and dynamic information
        relaxation: if constraint needs to be relaxed 
        ratio: missed sale to inventory cost ratio
        cost function: 
            "Hybrid": minimize inventory holding and missed sale cost, "Safety": get as close as possible to safety level 
            "Safety: imposes safety as constraint 
            "SafetyAddedHybrid: safety stocks are added to the hybrid solvers demand 
        constraint_coupling: 1:apply multi sku constraints, such as total inventory capacity for all the skus, 0: solve each sku separately  
        '''

        self.sku = range(skus.sku_count)
        #model = Model(solver_name="cbc")
        model = Model()
        self.cost_function = cost_function
        MS2IH = ratio
        # missed_sale_cost = [1,1, 1]
        # missed_sale_cost = missed_sale_cost
        self.order_limit = skus.constraints.supply_capacity
        self.inventory_capacity = skus.constraints.inventory_capacity
        self.n_levels = skus.n_levels
        self.time_step = 10
        self.time = list(range(self.time_step))
        self.level = list(range(self.n_levels))
        # order limits extracted from the article
        # purchase orders
        P = [[[model.add_var(var_type=CONTINUOUS, lb=0, ub=self.order_limit[j])
               for j in self.level] for i in self.time] for s in self.sku]

        # there is no inventory capacity constraint. Choosing a relatively large value
        # inventory held overnight,
        IH = [[[model.add_var(var_type=CONTINUOUS, lb=0, ub=self.inventory_capacity[j])
                for j in self.level] for i in self.time] for s in self.sku]

        # replenishment variable
        R = [[[model.add_var(var_type=CONTINUOUS, lb=0, ub=self.order_limit[j])
               for j in self.level] for i in self.time] for s in self.sku]

        MS = [[[model.add_var(var_type=CONTINUOUS, lb=0, ub=self.order_limit[j])
                for j in self.level] for i in self.time] for s in self.sku]

        if self.cost_function == "Safety":
            model.objective = minimize(xsum(IH[s][i][j]
                                            for i in self.time for j in self.level for s in self.sku))

        if self.cost_function == "Hybrid" or self.cost_function == "SafetyAddedHybrid":
            missed_sale_cost = []
            inv_cost = []
            # recreating missed sale and inventory cost with respect to the ratio.
            for s in self.sku:
                missed_sale_cost.append(skus.info.generic[s].missed_sale_cost)
                inv_cost.append(skus.info.generic[s].storage_cost)

            model.objective = minimize(xsum(inv_cost[s][j]*IH[s][i][j]
                                            for i in self.time for j in self.level for s in self.sku)
                                       + xsum(missed_sale_cost[s][j]*MS[s][i][j] for i in self.time for j in self.level for s in self.sku))

        if constraint_coupling == 1:
            # apply constraints according to constraints data type
            leads = list(skus.info.dynamic[0].leads)
            for i in self.time:
                for j in self.level:
                    if i > leads[j]:
                        model += xsum(IH[s][i][j] for s in self.sku) + xsum(R[s][i][j]
                                                                            for s in self.sku) <= skus.constraints.inventory_capacity[j]
            print('imposing inventory constraint:',
                  skus.constraints.inventory_capacity[j])
            # import time
            # time.sleep(3)

            # constraint R (replenishment) to purchase order according to lead
            # replenishment is equal to order placed with a lag = lead time for that stage
        for s in self.sku:
            transit_order = list(skus.info.dynamic[s].transit_order)
            leads = list(skus.info.dynamic[s].leads)
            for i in self.time:
                for j in self.level:
                    if i >= leads[j]:
                        model += R[s][i][j] == P[s][i-leads[j]][j]
                    else:
                        # -1 for compensate for zero indexing
                        model += R[s][i][j] == transit_order[j][leads[j]-i-1]

        # apply constraints for inventory holding for three levels of constraint rigidness
        # apply constraints for inventory holding constraint
        for s in self.sku:
            safety = list(skus.info.dynamic[s].safety)
            leads = list(skus.info.dynamic[s].leads)
            if cost_function == "SafetyAddedHybrid":
                D1 = list(skus.info.dynamic[s].iteration_demand_forecast)
                # add safety stock levels to current demand
                D = [demand+skus.info.dynamic[s].safety[0] for demand in D1]
            else:
                D = list(skus.info.dynamic[s].iteration_demand_forecast)

            inventory_holding = skus.info.dynamic[s].inventory
            transit_order = list(skus.info.dynamic[s].transit_order)
            for i in self.time:
                for j in self.level:
                    if i >= leads[j]:
                        if j < 2:
                            model += R[s][i][j] == P[s][i-leads[j]
                                                        ][j] - MS[s][i-leads[j]][j+1]
                        else:
                            # level 2 has infinite supply
                            model += R[s][i][j] == P[s][i-leads[j]][j]
                    else:
                        # -1 for compensate for zero indexing
                        model += R[s][i][j] == transit_order[j][leads[j]-i-1]
                for i in self.time:
                    for j in self.level:
                        if i == 0:
                            if j == 0:
                                model += inventory_holding[j] - IH[s][i][j] + \
                                    R[s][i][j] + MS[s][i][j] == D[i]
                                model += MS[s][i][j] >= D[i] - \
                                    inventory_holding[j] - R[s][i][j]
                                model += IH[s][i][j] >= - \
                                    (D[i]-inventory_holding[j] - R[s][i][j])
                                model += IH[s][i][j] >= 0
                            else:
                                model += inventory_holding[j] - IH[s][i][j] + \
                                    R[s][i][j] + MS[s][i][j] == P[s][i][j-1]
                                model += MS[s][i][j] >= P[s][i][j-1] - \
                                    inventory_holding[j] - R[s][i][j]
                                model += IH[s][i][j] >= - \
                                    (P[s][i][j-1] -
                                     inventory_holding[j] - R[s][i][j])
                                model += IH[s][i][j] >= 0

                        else:
                            if j == 0:
                                model += IH[s][i-1][j] - IH[s][i][j] + \
                                    R[s][i][j] + MS[s][i][j] == D[i]
                                model += MS[s][i][j] >= D[i] - \
                                    IH[s][i-1][j] - R[s][i][j]
                                model += IH[s][i][j] >= - \
                                    (D[i]-IH[s][i-1][j] - R[s][i][j])
                                model += IH[s][i][j] >= 0

                            else:
                                model += IH[s][i-1][j] - IH[s][i][j] + \
                                    R[s][i][j] + MS[s][i][j] == P[s][i][j-1]
                                model += MS[s][i][j] >= P[s][i][j -
                                                                1] - IH[s][i-1][j] - R[s][i][j]
                                model += IH[s][i][j] >= - \
                                    (P[s][i][j-1] - IH[s][i-1][j] - R[s][i][j])
                                model += IH[s][i][j] >= 0
                        if self.cost_function == "Safety":
                            # no missed sale allowed (note need relaxation for t<lead time )
                            if i > leads[j]:
                                model += MS[s][i][j] == 0
                            else:
                                model += MS[s][i][j] >= 0
                        else:
                            model += MS[s][i][j] >= 0

                        if self.cost_function == "Safety":
                            # inventory holding = safety stock level
                            if i >= leads[j]:
                                model += IH[s][i][j] >= safety[j]
                            else:
                                pass

        model.optimize(relax=True, max_seconds=100)
        # model.write('model.lp')
        # import time as tt
        # tt.sleep(5)
        print('<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<->>>>>>>>>>>>>>>>>>>>>>>>>>> "\n')

        for s in self.sku:
            print(f'------------<<<<<<<<sku{s}>>>>>>>-------------------:\n')
            print(f'lead time for sku{s} is {skus.info.dynamic[s].leads}')
            print(f"level zero sku{s}: ...................................:\n")

            if model.num_solutions:
                print("Solution with cost {} found.".format(
                    model.objective_value))

                print(
                    f'previous inventory holding:                              {inventory_holding[0]}')
                table = zip([P[s][i][0].x for i in self.time], [D[i] for i in self.time], [
                            R[s][i][0].x for i in self.time], [IH[s][i][0].x for i in self.time], [MS[s][i][0].x for i in self.time])
                print(tabulate(table, headers=['purchase order', 'demand',
                                               'replenishment', 'inventory holding', 'missed sale'], floatfmt=".4f"))
            else:
                print(f'Solution not found for sku{s}')

            print(f"level 1 sku{s}: ......................................:\n")
            if model.num_solutions:
                print(
                    f'previous inventory holding:                              {inventory_holding[1]}')
                table = zip([P[s][i][1].x for i in self.time], [P[s][i][0] for i in self.time], [
                            R[s][i][1].x for i in self.time], [IH[s][i][1].x for i in self.time], [MS[s][i][1].x for i in self.time])
                print(tabulate(table, headers=['purchase order', 'demand',
                                               'replenishment', 'inventory holding', 'missed sale'], floatfmt=".4f"))

            print(f"level 2 sku{s}: ......................................:\n")
            if model.num_solutions:
                print(
                    f'previous inventory holding:                              {inventory_holding[2]}')
                table = zip([P[s][i][2].x for i in self.time], [P[s][i][1] for i in self.time], [
                            R[s][i][2].x for i in self.time], [IH[s][i][2].x for i in self.time], [MS[s][i][2].x for i in self.time])
                print(tabulate(table, headers=['purchase order', 'demand',
                                               'replenishment', 'inventory holding', 'missed sale'], floatfmt=".4f"))
            print(model)
        if self.cost_function == "Safety":
            print(f'safety intended to be {safety}')
        try:
            print('sum of inventory at time 3 at level 1:', xsum(
                IH[s][3][1] for s in self.sku).x + xsum(R[s][3][1] for s in self.sku).x)
        except:
            print('no solution found')
        return [[P[s][0][j].x for j in self.level]for s in self.sku], [[MS[s][0][j].x for j in self.level] for s in self.sku], [[IH[s][0][j].x for j in self.level] for s in self.sku]

    @calc_runtime
    def get_mip_action_multiple_sku_v2(self, skus: SKUInfoFactory, ratio: float = 0.1, cost_function: str = "Safety", constraint_coupling: bool = 1, relaxation=0):
        '''
        Input:SKU object containing generic and dynamic information
        relaxation: if constraint needs to be relaxed 
        ratio: missed sale to inventory cost ratio
        cost function: 
            "Hybrid": minimize inventory holding and missed sale cost, "Safety": get as close as possible to safety level 
            "Safety: imposes safety as constraint 
            "SafetyAddedHybrid: safety stocks are added to the hybrid solvers demand 
        constraint_coupling: 1:apply multi sku constraints, such as total inventory capacity for all the skus, 0: solve each sku separately  
        '''
        self.order_limit = skus.constraints.supply_capacity
        self.inventory_capacity = skus.constraints.inventory_capacity
        self.sku = range(skus.sku_count)

        if cost_function == "Safety":
            pass
        else:
            raise Exception("v2 supports safety cost function only")

        '''
        Input: demand forecasts: expects , transit orders, 
        safety stock, previous safety
        Output: re-order levels 
        Missed sale cost is ignored for Safety optimization: goal is to never miss sale
        '''
        model = Model()
        self.cost_function = cost_function
        M2IH = ratio
        self.n_levels = skus.n_levels
        self.time_step = 10
        self.time = list(range(self.time_step))
        self.level = list(range(self.n_levels))

        P = [[[model.add_var(var_type=CONTINUOUS, lb=0, ub=self.order_limit[j])
               for j in self.level] for i in self.time] for s in self.sku]

        # there is no inventory capacity constraint. Choosing a relatively large value
        # inventory held overnight,
        IH = [[[model.add_var(var_type=CONTINUOUS, lb=0, ub=self.inventory_capacity[j])
                for j in self.level] for i in self.time] for s in self.sku]

        # replenishment variable
        R = [[[model.add_var(var_type=CONTINUOUS, lb=0, ub=self.order_limit[j])
               for j in self.level] for i in self.time] for s in self.sku]

        if self.cost_function == "Safety":
            model.objective = minimize(xsum(IH[s][i][j]
                                            for i in self.time for j in self.level for s in self.sku))

        if constraint_coupling == 1:
            # apply constraints according to constraints data type
            leads = list(skus.info.dynamic[0].leads)
            for i in self.time:
                for j in self.level:
                    if i > leads[j]:
                        model += xsum(IH[s][i][j] for s in self.sku) + xsum(R[s][i][j]
                                                                            for s in self.sku) <= skus.constraints.inventory_capacity[j]
            print('imposing inventory constraint:',
                  skus.constraints.inventory_capacity[j])

        for s in self.sku:
            safety = list(skus.info.dynamic[s].safety)
            leads = list(skus.info.dynamic[s].leads)
            D = list(skus.info.dynamic[s].iteration_demand_forecast)
            # constraint R (replenishment) to purchase order according to lead
            # replenishment is equal to order placed with a lag = lead time for that stage
            inventory_holding = skus.info.dynamic[s].inventory
            transit_order = list(skus.info.dynamic[s].transit_order)
            for i in self.time:
                for j in self.level:
                    if i >= leads[j]:
                        model += R[s][i][j] == P[s][i-leads[j]][j]
                    else:
                        # -1 for compensate for zero indexing
                        model += R[s][i][j] == transit_order[j][leads[j]-i-1]

            # apply constraints for inventory holding for three levels of constraint rigidness
            if relaxation == 0:
                # apply constraints for inventory holding constraint
                for i in self.time:
                    for j in self.level:
                        if i >= leads[j]:
                            if j == 0:
                                # apply demand at level zero
                                model += IH[s][i][j] == (IH[s][i-1][j] + R[s][i]
                                                         [j] - D[i])
                                model += IH[s][i][j] == safety[j]
                            else:
                                # apply order of the previous level as demand
                                model += IH[s][i][j] == (IH[s][i-1][j] + R[s][i]
                                                         [j] - P[s][i][j-1])
                                model += IH[s][i][j] == safety[j]
                        if i < leads[j] and i > 0:
                            if j == 0:
                                # apply demand at level zero
                                model += IH[s][i][j] == (IH[s][i-1][j] + R[s][i]
                                                         [j] - D[i])
                                model += IH[s][i][j] >= 0
                            else:
                                # apply order of the previous level as demand
                                model += IH[s][i][j] == (IH[s][i-1][j] + R[s][i]
                                                         [j] - P[s][i][j-1])
                                model += IH[s][i][j] >= 0
                        if i == 0:
                            if j == 0:
                                model += IH[s][i][j] == (inventory_holding[j] +
                                                         R[s][i][j] - D[i])
                                model += IH[s][i][j] >= 0  # safety[j]
                                # model += IH[i][j]
                            else:
                                model += IH[s][i][j] == (inventory_holding[j] +
                                                         R[s][i][j] - P[s][i][j-1])
                                model += IH[s][i][j] >= 0  # safety[j]
            elif relaxation == 1:
                for i in self.time:
                    for j in self.level:
                        if i >= leads[j]:
                            if j == 0:
                                # apply demand at level zero
                                model += IH[s][i][j] == (IH[s][i-1][j] + R[s][i]
                                                         [j] - D[i])
                                model += IH[s][i][j] == safety[j]
                            else:
                                # apply order of the previous level as demand
                                model += IH[s][i][j] == (IH[s][i-1][j] + R[s][i]
                                                         [j] - P[s][i][j-1])
                                model += IH[s][i][j] == safety[j]
                        if i < leads[j] and i > 0:
                            if j == 0:
                                # apply demand at level zero
                                model += IH[s][i][j] >= (IH[s][i-1][j] + R[s][i]
                                                         [j] - D[i])
                                model += IH[s][i][j] >= 0
                            else:
                                # apply order of the previous level as demand
                                model += IH[s][i][j] >= (IH[s][i-1][j] + R[s][i]
                                                         [j] - P[s][i][j-1])
                                model += IH[s][i][j] >= 0
                        if i == 0:
                            if j == 0:
                                model += IH[s][i][j] >= (inventory_holding[j] +
                                                         R[s][i][j] - D[i])
                                model += IH[s][i][j] >= 0  # safety[j]
                                # model += IH[i][j]
                            else:
                                model += IH[s][i][j] >= (inventory_holding[j] +
                                                         R[s][i][j] - P[s][i][j-1])
                                model += IH[s][i][j] >= 0  # safety[j]
            elif relaxation == 2:
                # apply constraints for inventory holding constraint
                for i in self.time:
                    for j in self.level:
                        if i >= leads[j]:
                            if j == 0:
                                # apply demand at level zero
                                model += IH[s][i][j] == (IH[s][i-1][j] + R[s][i]
                                                         [j] - D[i])
                                model += IH[s][i][j] >= safety[j]
                            else:
                                # apply order of the previous level as demand
                                model += IH[s][i][j] == (IH[s][i-1][j] + R[s][i]
                                                         [j] - P[s][i][j-1])
                                model += IH[s][i][j] >= safety[j]
                        if i < leads[j] and i > 0:
                            if j == 0:
                                # apply demand at level zero
                                model += IH[s][i][j] >= (IH[s][i-1][j] + R[s][i]
                                                         [j] - D[i])
                                model += IH[s][i][j] >= 0
                            else:
                                # apply order of the previous level as demand
                                model += IH[s][i][j] >= (IH[s][i-1]
                                                         [j] + R[s][i][j] - P[s][i][j-1])
                                model += IH[s][i][j] >= 0
                        if i == 0:
                            if j == 0:
                                model += IH[s][i][j] >= (inventory_holding[j] +
                                                         R[s][i][j] - D[i])
                                model += IH[s][i][j] >= 0  # safety[j]
                                # model += IH[i][j]
                            else:
                                model += IH[s][i][j] >= (inventory_holding[j] +
                                                         R[s][i][j] - P[s][i][j-1])
                                model += IH[s][i][j] >= 0  # safety[j]

        model.optimize(relax=True, max_seconds=200)

        # calculate projected missed sale
        MS = np.empty((len(self.sku), len(self.time),
                      len(self.level)), dtype=int)
        try:
            for s in self.sku:
                inventory_holding = skus.info.dynamic[s].inventory
                D = list(skus.info.dynamic[s].iteration_demand_forecast)
                for i in self.time:
                    for j in self.level:
                        if i == 0:
                            if j == 0:
                                MS[s, i, j] = abs(min(0, (inventory_holding[j] +
                                                          R[s][i][j].x - D[i])))
                            else:
                                MS[s, i, j] = abs(min(0, inventory_holding[j] +
                                                      R[s][i][j].x - P[s][i][j-1].x))
                        else:
                            if j == 0:
                                MS[s][i][j] = abs(min(0, IH[s][i-1][j].x + R[s][i]
                                                      [j].x - D[i]))
                            else:
                                MS[s][i, j] = abs(min(0, IH[s][i-1][j].x + R[s][i]
                                                      [j].x - P[s][i][j-1].x))
        except:
            print('warning: MSL not calculated ...')

        #     print("level zero: ...................................:\n")

        #     if model.num_solutions:
        #         print("Solution with cost {} found.".format(model.objective_value))

        #         print(
        #             f'previous inventory holding:                              {inventory_holding[0]}')
        #         table = zip([P[i][0].x for i in self.time], [D[i] for i in self.time], [
        #                     R[i][0].x for i in self.time], [IH[i][0].x for i in self.time],[MSL[i][0] for i in self.time])
        #         print(tabulate(table, headers=['purchase order', 'demand',
        #                                     'replenishment', 'inventory holding', 'missed sale'], floatfmt=".4f"))

        #     print("level 1: ......................................:\n")
        #     if model.num_solutions:
        #         print(
        #             f'previous inventory holding:                              {inventory_holding[1]}')
        #         table = zip([P[i][1].x for i in self.time], [P[i][0] for i in self.time], [
        #                     R[i][1].x for i in self.time], [IH[i][1].x for i in self.time],[MSL[i][1] for i in self.time])
        #         print(tabulate(table, headers=['purchase order', 'demand',
        #                                     'replenishment', 'inventory holding', 'missed sale'], floatfmt=".4f"))

        #     print("level 2: ......................................:\n")
        #     if model.num_solutions:
        #         print(
        #             f'previous inventory holding:                              {inventory_holding[2]}')
        #         table = zip([P[i][2].x for i in self.time], [P[i][1] for i in self.time], [
        #                     R[i][2].x for i in self.time], [IH[i][2].x for i in self.time],[MSL[i][2] for i in self.time])
        #         print(tabulate(table, headers=['purchase order', 'demand',
        #                                     'replenishment', 'inventory holding', 'missed sale'], floatfmt=".4f"))
        #     print(model.status)
        #     print(f'safety intended to be {safety}')
        #     return [P[0][j].x for j in self.level], [[MSL[i][j] for i in range(0,5)] for j in self.level], [[IH[i][j].x for i in range(0,5)] for j in self.level]
        # model.optimize(relax = True, max_seconds = 100)
        # # model.write('model.lp')
        # # import time as tt
        # # tt.sleep(5)
        print('<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<->>>>>>>>>>>>>>>>>>>>>>>>>>> "\n')

        for s in self.sku:
            print(f'------------<<<<<<<<sku{s}>>>>>>>-------------------:\n')
            print(f'lead time for sku{s} is {skus.info.dynamic[s].leads}')
            print(f"level zero sku{s}: ...................................:\n")

            if model.num_solutions:
                print("Solution with cost {} found.".format(
                    model.objective_value))

                print(
                    f'previous inventory holding:                              {inventory_holding[0]}')
                table = zip([P[s][i][0].x for i in self.time], [D[i] for i in self.time], [
                            R[s][i][0].x for i in self.time], [IH[s][i][0].x for i in self.time], [MS[s][i][0] for i in self.time])
                print(tabulate(table, headers=['purchase order', 'demand',
                                               'replenishment', 'inventory holding', 'missed sale'], floatfmt=".4f"))
            else:
                print(f'Solution not found for sku{s}')

            print(f"level 1 sku{s}: ......................................:\n")
            if model.num_solutions:
                print(
                    f'previous inventory holding:                              {inventory_holding[1]}')
                table = zip([P[s][i][1].x for i in self.time], [P[s][i][0] for i in self.time], [
                            R[s][i][1].x for i in self.time], [IH[s][i][1].x for i in self.time], [MS[s][i][1] for i in self.time])
                print(tabulate(table, headers=['purchase order', 'demand',
                                               'replenishment', 'inventory holding', 'missed sale'], floatfmt=".4f"))

            print(f"level 2 sku{s}: ......................................:\n")
            if model.num_solutions:
                print(
                    f'previous inventory holding:                              {inventory_holding[2]}')
                table = zip([P[s][i][2].x for i in self.time], [P[s][i][1] for i in self.time], [
                            R[s][i][2].x for i in self.time], [IH[s][i][2].x for i in self.time], [MS[s][i][2] for i in self.time])
                print(tabulate(table, headers=['purchase order', 'demand',
                                               'replenishment', 'inventory holding', 'missed sale'], floatfmt=".4f"))
            print(model)
        if self.cost_function == "Safety":
            print(f'safety intended to be {safety}')
        try:
            print('sum of inventory at time 3 at level 1:', xsum(
                IH[s][3][1] for s in self.sku).x + xsum(R[s][3][1] for s in self.sku).x)
        except:
            print('no solution found')
        return [[P[s][0][j].x for j in self.level]for s in self.sku], [[MS[s][0][j] for j in self.level] for s in self.sku], [[IH[s][0][j].x for j in self.level] for s in self.sku]


if __name__ == "__main__":
    n_sku = 2
    n_stages = 4
    test_chain = SupplyChainTopology(number_of_stages=n_stages)
    test_skus = SKUInfoFactory(sku_count=n_sku, topology=test_chain)
    test_skus.constraints.inventory_capacity = [50, 50, 50]
    my_solver = MipSolver(config=None)
    po, ms, ih = my_solver.get_mip_action_multiple_sku_v2(
        skus=test_skus, cost_function="Safety", constraint_coupling=1, relaxation=1)
    print('purchase order:\n', po)
    print('inventory_capacity:\n', test_skus.constraints.inventory_capacity)
    print(test_skus.info.dynamic[0].safety)
    print(test_skus.info.dynamic[1].safety)
