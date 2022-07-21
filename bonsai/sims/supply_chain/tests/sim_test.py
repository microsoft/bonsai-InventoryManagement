'''
Set of tests to make sure multi_sku sim works as intended 
'''
from supply_chain.inventory_management import InvManagementLostSalesMultiSKUEnv
from supply_chain.multi_sku import SKUInfoFactory
from supply_chain.chain_definition import SupplyChainTopology
import pytest
import random
import numpy as np


@pytest.fixture
def SKUS() -> SKUInfoFactory:
    n_sku = random.randint(2, 300)
    n_stages = random.randint(3, 7)
    chain = SupplyChainTopology(number_of_stages=n_stages)
    return SKUInfoFactory(sku_count=n_sku, topology=chain)


def test_end_to_end_episode(SKUS: SKUInfoFactory) -> None:
    '''
    make sure env.steps run for an episode length of 200 against an arbitrary number of skus and stages
    '''
    episode_length = 200
    env = InvManagementLostSalesMultiSKUEnv(skus=SKUS)
    output = env.reset()
    for step in range(episode_length):
        random_action = np.random.randint(
            10, 200, (SKUS.sku_count, SKUS.n_levels))
        states = env.step(random_action)


def test_episode_transit_order_iteration0(SKUS: SKUInfoFactory) -> None:
    env = InvManagementLostSalesMultiSKUEnv(skus=SKUS)
    env.reset()
    fixed_action = np.random.randint(10, 100, (SKUS.sku_count, SKUS.n_levels))
    env.step(fixed_action)
    for s in range(SKUS.sku_count):
        for l in range(SKUS.n_levels):
            assert env.episode_transit_orders[s,
                                              env.current_lead_time[s, l], l] == fixed_action[s, l]


def test_espisode_transit_orders_iteration_n(SKUS: SKUInfoFactory) -> None:
    '''
    start and episode with zero actions, then impose a fixed action 
    '''
    env = InvManagementLostSalesMultiSKUEnv(skus=SKUS)
    env.reset()
    n = 33  # random
    zero_action = np.random.randint(0, 1, (SKUS.sku_count, SKUS.n_levels))
    for i in range(n):
        env.step(zero_action)
    fixed_action = np.random.randint(10, 100, (SKUS.sku_count, SKUS.n_levels))
    env.step(fixed_action)
    for s in range(SKUS.sku_count):
        for l in range(SKUS.n_levels):
            assert env.episode_transit_orders[s, n +
                                              env.current_lead_time[s, l], l] == fixed_action[s, l]
