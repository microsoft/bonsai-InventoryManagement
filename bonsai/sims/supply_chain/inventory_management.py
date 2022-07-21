'''
__author__: Hossein Khadivi Heris @microsoft 

Some parts of the code are inspired by and used from  
or-gym library by the following authors
Hector Perez, Christian Hubbs, Owais Sarwar
4/14/2020
- supported: for bonsai integration 
- not supported for gym integration. Hint: to support, following is needed: 
    modify gym observation statement in episode reset
    modify get sim state in episode step 
'''
from ast import Raise
import itertools
from typing import Any, Type, List
from scipy.stats import *
import numpy as np
import gym
from utils import assign_env_config
from utils import make_lead_profile
from collections import deque
from multi_sku import SKUInfoFactory
from chain_definition import SupplyChainTopology


class InvManagementMultiSKUMasterEnv(gym.Env):
    '''
    The supply chain environment is structured based on or-gym paper [https://arxiv.org/abs/2008.06319] as found in the quote below. 

    "It is a multi-period multi-echelon production-inventory system for non-perishable products that is sold only
    in discrete quantities. Each stage in the supply chain consists of an inventory holding area and a production area.
    The exception are the first stage (retailer: only inventory area) and the last stage (raw material transformation
    plant: only production area, with unlimited raw material availability). The inventory holding area holds the inventory
    necessary to produce the material at that stage. One unit of inventory produces one unit of product at each stage.
    There are lead times between the transfer of material from one stage to the next. The outgoing material from stage i
    is the feed material for production at stage i-1. Stages are numbered in ascending order: Stages = {0, 1, ..., M}
    (i.e. m = 0 is the retailer). Production at each stage is bounded by the stage's production capacity and the available
    inventory.

    At the beginning of each time period, the following sequence of events occurs:

    0) Stages 0 through M-1 place replenishment orders to their respective suppliers. Replenishment orders are filled
        according to available production capacity and available inventory at the respective suppliers.
    1) Stages 0 through M-1 receive incoming inventory replenishment shipments that have made it down the product pipeline
        after the stage's respective lead time.
    2) Customer demand occurs at stage 0 (retailer). It is sampled from a specified discrete probability distribution.
    3) Demand is filled according to available inventory at stage 0.
    4) Unfulfilled sales and replenishment orders are lost with a goodwill loss penalty.
        a) Hint: you may extend the sim for backlog
            Unfulfilled sales and replenishment orders are backlogged at a penalty.
            Note: Backlogged sales take priority in the following period
    5) Surpluss inventory is held at each stage at a holding cost."

    The following properties are added to the simulator:
    6) Multiple SKUs is accepted by the simulator through multisku object with 
        a) Custom demand profile for each SKU with the following properties being parametrized 
            - non-stationary 
            - noisy
            - uncertain 
        b) different lead profile for each sku          
    '''

    def __init__(self, skus: SKUInfoFactory = SKUInfoFactory(sku_count=10), *args, **kwargs):
        '''
        sku object contains information needed 
        '''
        # set default (arbitrary) values when creating environment (if no args or kwargs are given)
        self.skus = skus
        self.n_sku = skus.sku_count
        self.num_stages = skus.n_levels + 1
        self.periods = 365
        self.supply_capacity = skus.constraints.supply_capacity

        self._rearrange_sku_data()

        self.total_sku_constraint = True  # applicable to multi-sku

        self.dist = 5  # it is a custom demand
        self.backlog = True
        self.seed_int = 0
        self._max_rewards = 2000

        # add/overwrite environment configuration dictionary and keyword arguments
        assign_env_config(self, kwargs)

        self.num_periods = self.periods

        self.reset()

        self.pipeline_length = (
            self.n_sku)*(self.num_stages - 1)*(self.max_transit_order_tracking_days + 1)

        self.action_space = gym.spaces.Box(
            low=0, high=100, shape=(self.n_sku, self.num_stages - 1), dtype=np.int16)
        # observation space (Inventory position at each echelon, which is any integer value)

        # observation space will be ignored and instead specific states will be used for brain training
        # note preserving the order for the original sim but with more
        self.observation_space = gym.spaces.Box(
            low=-10,
            high=100,
            # +1 to include inventory (s, 0, :) level and the rest (s,1:, :) include transit orders
            shape=(self.n_sku, self.max_transit_order_tracking_days + 1,
                   self.num_stages-1),  # [sku[inv+stages [transit orders]]]
            # example [[I0 .. I2, R*3stages*15tracking][inv0, inve]]
            dtype=np.int32)

    def _rearrange_sku_data(self):
        '''
        construct 3D matrix from sku object
        '''
        skus = self.skus
        self.initial_inventory = []
        self.lead_profiles = []
        self.user_demand = []
        self.transit_orders = []
        self.unit_prices = []
        self.unit_costs = []
        self.holding_costs = []  # inventory cost
        self.missed_sale_cost = []
        # values =value [sku, level]
        for s in range(skus.sku_count):
            self.initial_inventory.append(skus.info.dynamic[s].inventory)
            self.lead_profiles.append(skus.info.dynamic[s].lead_profile)
            self.user_demand.append(
                skus.info.dynamic[s].demand_info.demand_actual)
            self.transit_orders.append(skus.info.dynamic[s].transit_order)
            self.unit_prices.append(skus.info.generic[s].unit_price[:-1])
            self.unit_costs.append(skus.info.generic[s].unit_cost[:-1])
            self.holding_costs.append(skus.info.generic[s].storage_cost[:-1])
            self.missed_sale_cost.append(
                skus.info.generic[s].missed_sale_cost[:-1])

        self.unit_prices = np.array(self.unit_prices)
        self.unit_costs = np.array(self.unit_costs)
        self.holding_costs = np.array(self.holding_costs)
        self.missed_sale_cost = np.array(self.missed_sale_cost)
        self.user_D = np.array(self.user_demand)
        # reduced tracking all the transit order above day 14 are squeeze into day 15.
        self.transit_order_tracking_days = 15
        # total tracking of the transit orders in days. Usually 30 days is sufficient for leads below 10 days
        self.max_transit_order_tracking_days = 30
        # transit orders kept up to max_transit_order_tracking days
        self.state_in_transit = np.zeros(
            (self.n_sku, self.num_stages-1, self.max_transit_order_tracking_days))  # [45*[0]]
        self.lead_profiles = np.array(self.lead_profiles)

    def seed(self, seed=None):
        '''
        Set random number generation seed
        '''
        # seed random state
        if seed != None:
            np.random.seed(seed=int(seed))

    def _RESET(self):
        '''
        Create variables and initialize numpy variables for matrix calculations  
        '''
        periods = self.num_periods
        m = self.num_stages
        n_sku = self.n_sku
        self.Inventory = np.zeros([n_sku, periods + 1, m - 1])
        # pipeline inventory at the beginning of each period (no pipeline inventory for last stage)
        self.Replenishment = np.zeros([n_sku, periods, m - 1])  # + 25
        self.Demand = np.zeros([n_sku, periods])  # demand at retailer
        self.Sold_unit = np.zeros([n_sku, periods, m-1])  # units sold
        # backlog (includes top most production site in supply chain)
        self.Backlog = np.zeros([n_sku, periods, m-1])
        self.LostSale = np.zeros([n_sku, periods, m-1])  # lost sales
        self.ProfitPerSku = np.zeros([n_sku, periods])  # profit per SKU
        self.ProfitSum = np.zeros([periods])  # total profit
        # missed sale and inventory cost per sku per time
        self.CostPerSku = np.zeros([n_sku, periods])
        # missed sale and inventory cost for all skus per time
        self.CostTotal = np.zeros([periods])

        # initialization
        self.period = 0  # initialize time
        self.Inventory[:, self.period, :] = np.array(
            self.initial_inventory)  # initial inventory

        # attn: note that periods come first as rl actions will be in the format of [[actions_sku0]...[actions_sku1]]
        self.action_log = np.zeros((self.n_sku, periods+1, m-1))
        self.episode_transit_orders = np.zeros(
            (self.n_sku, self.periods+1, m-1))

        self._update_state()

        return self.state

    def _update_state(self):
        m = self.num_stages - 1
        t = self.period
        lt_max = self.transit_order_tracking_days
        state = np.zeros((self.n_sku, (lt_max + 1), m))

        if t == 0:
            state[:, 0, :] = self.initial_inventory
            self.current_inventory_level = np.array(self.initial_inventory)
        else:
            # 2D array [sku, level] will reference
            self.current_inventory_level = np.array(self.Inventory[:, t, :])
            # 3D array, one can access current inventory level through state[s, 0, level]
            state[:, 0, :] = self.Inventory[:, t, :]

        # previous orders, the last state contains sum of all
        if t == 0:
            pass
        elif t >= lt_max:
            state[:, 1:lt_max, :] = self.action_log[:, t-lt_max+1:t, :]
        else:
            state[:, 1:t, :] = self.action_log[:, 1:t, :]

        self.state = state.copy()

    def _STEP(self, action: List[List[int]]) -> Any:
        '''
        Take a step in time in the multi-period multi skus inventory management problem.
        action shape = (sku, levels) number of units to request from suppliers (last stage makes no requests)
        [orders_sku0 [10,20,30], orders_sku1[30, 20,10]]
        '''
        replenish_action = np.maximum(action, 0).astype(int)

        # lead time across all the
        self.current_lead_time = self.lead_profiles[:, :, self.period]
        current_supply_capacity = np.array(
            self.n_sku*[self.skus.constraints.supply_capacity])  # capacity

        # check_supply_capacity_constraint
        replenish_action = np.minimum(
            replenish_action, current_supply_capacity)

        # add replenishment to in_transit_order for future date
        for sku in range(self.n_sku):
            for level in range(self.num_stages-1):
                # update only if orders are within episode length (prevent error)
                if (self.period + self.current_lead_time[sku, level]) < (self.periods-1):
                    self.episode_transit_orders[sku, self.period +
                                                self.current_lead_time[sku, level], level] = replenish_action[sku, level]

        self.iteration_replenishment = self.episode_transit_orders[:,
                                                                   self.period, :]
        # update_inventory for current period
        if self.period > 0:
            self.Inventory[:, self.period, :] = self.Inventory[:,
                                                               self.period-1, :] + self.episode_transit_orders[:, self.period, :]
        else:
            pass

        self.action_log[:, self.period, :] = replenish_action

        print('In transit order for the whole episode :\n',
              self.episode_transit_orders)

        # state specifically designed to track transit order up to max_transit_order days, more complete info than gym state
        self.state_in_transit = np.zeros(
            (self.n_sku, self.max_transit_order_tracking_days, self.num_stages-1))  # [45*[0]]

        self.state_in_transit[:, 0:self.max_transit_order_tracking_days - 1, :] = self.episode_transit_orders[:,
                                                                                                              self.period + 1:self.period + self.max_transit_order_tracking_days, :]
        self.state_in_transit[:, self.max_transit_order_tracking_days - 1, :] = np.sum(
            self.episode_transit_orders[:, self.period + self.max_transit_order_tracking_days:], axis=1)

        # demand is realized
        if self.dist < 5:
            raise Exception("demand type other than custom demand\
                is not supported for multi-sku environment")
            D = [self.demand_dist.rvs(**self.dist_param)
                 for s in range(self.n_sku)]
        else:
            D = self.user_D[:, self.period].reshape(
                self.n_sku, -1)  # user specified demand

        # # add previous backlog to demand
        # if n >= 1:
        #     D = D + self.Backlog[:, self.period-1, 0].copy()  # add backlogs to demand

        # units sold
        self.Demand_all_levels = np.append(
            D[:], replenish_action[:, :-1], axis=1)
        self.Sold_unit[:, self.period, :] = np.minimum(
            self.Inventory[:, self.period, :], self.Demand_all_levels)
        # self.Sold_unit[:,self.period, 0] = np.minimum(self.Inventory[:,self.period,0].reshape(self.n_sku,-1),D[:,self.period].reshape(self.n_sku, -1))
        # self.Sold_unit[:, self.period, 1:] = np.minimum(self.Inventory[:,self.period,1:].reshape(self.n_sku, -1), replenish_action[:, :-1].reshape(self.n_sku, -1))

        self.Inventory[:, self.period, :] = self.Inventory[:,
                                                           self.period, :] - self.Sold_unit[:, self.period, :]
        # update inventory on hand and pipeline inventory

        # unfulfilled demand and replenishment orders
        self.LostSale[:, self.period, :] = np.abs(
            self.Sold_unit[:, self.period, :]-self.Demand_all_levels)
        self.iteration_lost_sale = self.LostSale[:, self.period, :]
        self.unit_prices = np.array(self.unit_prices)
        self.unit_costs = np.array(self.unit_costs)
        self.holding_costs = np.array(self.holding_costs)
        self.missed_sale_cost = np.array(self.missed_sale_cost)
        self.ProfitPerSku[:, self.period] = np.sum(self.Sold_unit[:, self.period, :]*self.unit_prices -
                                                   self.unit_costs*replenish_action -
                                                   self.missed_sale_cost*self.LostSale[:, self.period, :] -
                                                   self.holding_costs*self.Inventory[:, self.period, :], axis=1)
        self.ProfitSum[self.period] = np.sum(self.Sold_unit[:, self.period, :]*self.unit_prices -
                                             self.unit_costs*replenish_action -
                                             self.missed_sale_cost*self.LostSale[:, self.period, :] -
                                             self.holding_costs *
                                             self.Inventory[:, self.period, :]
                                             )
        self.CostTotal[self.period] = np.sum(- self.missed_sale_cost*self.LostSale[:, self.period, :] -
                                             self.holding_costs*self.Inventory[:, self.period, :])
        self.CostPerSku[:, self.period] = np.sum(- self.missed_sale_cost*self.LostSale[:, self.period, :] -
                                                 self.holding_costs *
                                                 self.Inventory[:,
                                                                self.period, :],
                                                 axis=1)

        # update state
        self._update_state()
        self._update_skus_object()

        self.reward = self.CostTotal[self.period]
        self.iteration_cost = self.CostTotal[self.period]
        # determine if simulation should terminate
        if self.period >= self.num_periods - 1:
            done = True
        else:
            done = False
        self.period += 1
        self.skus.update_iteration_for_all_skus(iteration=self.period)
        self.skus.update_transit_orders_for_all_skus(
            iteration_purchase_orders=replenish_action.tolist())
        self.skus.update_lead_for_all_skus()
        self.skus.update_inventory_for_all_skus(
            inventory_levels=self.current_inventory_level.tolist())
        return self.state, self.reward, done, {}

    def _update_skus_object(self):
        '''
        Update inventory
        Update transit orders 
        '''
        for sku in range(self.n_sku):
            self.skus.info.dynamic[sku].inventory = self.Inventory[sku,
                                                                   self.period, :]

    def sample_action(self):
        '''
        Generate an action by sampling from the action_space
        '''
        return self.action_space.sample()

    def step(self, action):
        return self._STEP(action)

    def reset(self):
        return self._RESET()


class InvManagementLostSalesMultiSKUEnv(InvManagementMultiSKUMasterEnv):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.backlog = False
        self.observation_space = gym.spaces.Box(
            # Never goes negative without backlog
            low=np.zeros(self.pipeline_length),
            high=np.ones(self.pipeline_length)*max(self.supply_capacity)*self.num_periods, dtype=np.int32)


if __name__ == "__main__":
    n_sku = 5
    n_levels = 3
    my_chain = SupplyChainTopology(number_of_stages=n_levels+1)
    my_config = {'offset': 40}
    skus = SKUInfoFactory(
        sku_count=n_sku, topology=my_chain, config=my_config)
    env = InvManagementLostSalesMultiSKUEnv(skus=skus)
    output = env.reset()
    rand_act = np.random.randint(1, 100, (n_sku, n_levels))
    for i in range(100):
        states = env.step(rand_act)
        skus.update_transit_orders_for_all_skus(rand_act)
