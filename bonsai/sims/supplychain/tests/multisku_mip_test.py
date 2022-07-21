import random
from supplychain.multi_sku import SKUInfoFactory
from supplychain.multi_sku import SupplyChainTopology
from supplychain.mip_solver import MipSolver
import pytest
from collections import deque


@pytest.fixture
def Solver() -> MipSolver:
    return MipSolver(config=None)


def test_solver_run() -> None:
    n_sku = 50
    n_stages = 4
    test_chain = SupplyChainTopology(number_of_stages=n_stages)
    test_skus = SKUInfoFactory(sku_count=n_sku, topology=test_chain)
    solver = MipSolver(config=None)
    solver.get_mip_action_multiple_sku(skus=test_skus)


def test_mip_solver_hybrid_n_stage0() -> None:
    n_sku = 88
    n_stages = 5
    test_chain = SupplyChainTopology(number_of_stages=n_stages)
    test_skus = SKUInfoFactory(sku_count=n_sku, topology=test_chain)
    solver = MipSolver(config=None)
    solver.get_mip_action_multiple_sku(skus=test_skus)


def test_solver_run_all_cost_functions() -> None:
    '''
    test to make sure all three cost functions can run with multisku object 
    '''
    n_sku = 31
    n_stages = 4
    test_chain = SupplyChainTopology(number_of_stages=n_stages)
    test_skus = SKUInfoFactory(sku_count=n_sku, topology=test_chain)
    solver = MipSolver(config=None)
    test_skus.update_iteration_for_all_skus(iteration=random.randint(0, 100))
    solver.get_mip_action_multiple_sku(skus=test_skus, cost_function="Hybrid")
    solver.get_mip_action_multiple_sku(skus=test_skus, cost_function="Safety")
    solver.get_mip_action_multiple_sku(
        skus=test_skus, cost_function="SafetyAddedHybrid")


def test_solver_run_specific(Solver: MipSolver) -> None:
    import pickle
    with open('FailedSkuObject1.pickle', 'rb') as handle:
        skus = pickle.load(handle)
    PO1, _, _ = Solver.get_mip_action_multiple_sku(
        skus=skus, cost_function="Hybrid")
    PO2, _, _ = Solver.get_mip_action_multiple_sku(
        skus=skus, cost_function="SafetyAddedHybrid")
    assert None not in [item for sublist in PO1 for item in sublist]
    assert None not in [item for sublist in PO1 for item in sublist]


def test_solver_fail_capacity(Solver: MipSolver) -> None:
    import pickle
    with open('FailedSkuObject2.pickle', 'rb') as handle:
        skus = pickle.load(handle)
    skus.info.dynamic[0].transit_order = [deque([0, 20, 20, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], maxlen=45),
                                          deque([0, 20, 20, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], maxlen=45),
                                          deque([0, 20, 20, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], maxlen=45)]
    skus.info.dynamic[0].inventory = [1, 3, 2]
    skus.info.dynamic[0].safety = [5.0, 5.0, 6.0]
    skus.info.dynamic[0].leads = [2, 3, 4]
    skus.constraints.inventory_capacity = [20, 20, 20]
    PO1, _, _ = Solver.get_mip_action_multiple_sku(
        skus=skus, cost_function="Hybrid")
    PO2, _, _ = Solver.get_mip_action_multiple_sku(
        skus=skus, cost_function="SafetyAddedHybrid")
    assert None in [item for sublist in PO1 for item in sublist]
    assert None in [item for sublist in PO2 for item in sublist]
