"""
__author__: Hossein Khadivi Heris@microsoft 
type declaration for constraints 
"""
from dataclasses import dataclass, field, asdict, replace
from typing import Union, List, Deque, Dict
from .chain_definition import SupplyChainTopology


def default_factory_inventory_capacity():
    sc = SupplyChainTopology()
    n_levels = sc.number_of_stages - 1  # last stage has infinite supply and capacity
    return n_levels*[200]


def default_factory_supply_capacity():
    sc = SupplyChainTopology()
    n_levels = sc.number_of_stages - 1  # last stage has infinite supply and capacity
    # [100, 90, 80, etc ..]
    return [(100-10*i) for i in range(n_levels)]


@dataclass
class MultiSKUConstraints:
    # a list with size of number of levels in the multi-echelon inventory management
    inventory_capacity: List[int] = field(
        init=True, default_factory=default_factory_inventory_capacity)
    # max order to be handled over all the skus at single sku level. i.e. order<supply_capacity
    supply_capacity: List[int] = field(
        init=True, default_factory=default_factory_supply_capacity)
    # however, can be used to impose as max truck capacity when supplying multisku from different level
