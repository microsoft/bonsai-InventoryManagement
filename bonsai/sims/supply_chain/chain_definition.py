"""Class defines the topology of the supply chain 
 """

from dataclasses import dataclass, field
from typing import List


@dataclass
class SupplyChainTopology:
    number_of_stages: int = field(default=4)
