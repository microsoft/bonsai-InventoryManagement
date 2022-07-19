from random import randint
from tkinter import N
from or_gym.envs.supply_chain.multi_sku import *
from or_gym.envs.supply_chain.chain_definition import SupplyChainTopology
import pytest


@pytest.fixture
def SKUS() -> SKUInfoFactory:
    n_sku = randint(10, 200)
    return SKUInfoFactory(sku_count=n_sku)


def test_sku_count() -> None:
    n_sku = randint(10, 200)
    skus = SKUInfoFactory(sku_count=n_sku)
    assert skus.sku_count == n_sku


def test_sku_count_change(SKUS: SKUInfoFactory) -> None:
    n_sku = randint(10, 200)
    SKUS.sku_count = n_sku
    assert SKUS.sku_count == n_sku


def test_sku_price_dimension() -> None:
    n_sku = 10
    n_stages = [4, 5]
    for n_stage in n_stages:
        chain = SupplyChainTopology(number_of_stages=n_stage)
        skus = SKUInfoFactory(sku_count=n_sku, topography=chain)
        assert len(skus.info.generic[0].unit_cost) == n_stage


def test_sku_inventory_dimension() -> None:
    n_sku = 10
    n_stages = [4, 5]
    for n_stage in n_stages:
        chain = SupplyChainTopology(number_of_stages=n_stage)
        skus = SKUInfoFactory(sku_count=n_sku, topography=chain)
        assert len(skus.info.dynamic[0].inventory) == n_stage-1


def test_sku_lead_profile_dimension() -> None:
    n_sku = 10
    n_stages = [4, 5]
    for n_stage in n_stages:
        chain = SupplyChainTopology(number_of_stages=n_stage)
        skus = SKUInfoFactory(sku_count=n_sku, topography=chain)
        assert len(skus.info.dynamic[0].lead_profile) == n_stage-1


def test_sku_demand_profile() -> None:
    '''
    make sure each demand is unique
    Note: it might falsely fail with very small probability 
    '''
    n_sku = 10
    n_stages = [5]
    for n_stage in n_stages:
        chain = SupplyChainTopology(number_of_stages=n_stage)
        skus = SKUInfoFactory(sku_count=n_sku, topography=chain)
        assert (skus.info.dynamic[0].demand_info.demand_actual[0] != skus.info.dynamic[1].demand_info.demand_actual[0])\
            or skus.info.dynamic[0].demand_info.demand_actual[0] != skus.info.dynamic[2].demand_info.demand_actual[0]


def test_constraint_shape1() -> None:
    n_sku = 100
    n_stages = [4, 5]
    for n_stage in n_stages:
        chain = SupplyChainTopology(number_of_stages=n_stage)
        skus = SKUInfoFactory(sku_count=n_sku, topography=chain)
        assert len(skus.constraints.inventory_capacity) == n_stage-1


def test_constraint_shape2() -> None:
    n_sku = 100
    n_stages = [4, 5]
    for n_stage in n_stages:
        chain = SupplyChainTopology(number_of_stages=n_stage)
        skus = SKUInfoFactory(sku_count=n_sku, topography=chain)
        assert len(skus.constraints.supply_capacity) == n_stage-1


def test_demand_forecast_property() -> None:
    n_sku = 10
    n_stages = 4
    chain = SupplyChainTopology(number_of_stages=n_stages)
    skus = SKUInfoFactory(sku_count=n_sku, topography=chain)
    for s in range(skus.sku_count):
        skus.info.dynamic[s].period = 0
        forecast_iteration0 = skus.info.dynamic[s].iteration_demand_forecast
        skus.info.dynamic[s].period = 5
        forecast_iteration5 = skus.info.dynamic[s].iteration_demand_forecast
        assert forecast_iteration0[5] == forecast_iteration5[0]


def test_demand_forecast_length(SKUS: SKUInfoFactory) -> None:
    assert len(
        SKUS.info.dynamic[0].iteration_demand_forecast) == SKUS.info.dynamic[0].forecast_window


def test_transit_order_update() -> None:
    n_sku = 5
    n_stages = 4
    chain = SupplyChainTopology(number_of_stages=n_stages)
    skus = SKUInfoFactory(sku_count=n_sku, topography=chain)
    actions = [[8, 17, 0],
               [8, 23, 23],
               [8, 23, 23],
               [8, 22, 22],
               [8, 22, 19]]
    skus.update_transit_orders_for_all_skus(iteration_purchase_orders=actions)
    for sku in range(n_sku):
        for level in range(n_stages-1):
            assert skus.info.dynamic[sku].transit_order[level][0] == actions[sku][level]
