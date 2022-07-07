"""Class defines the topology of the supply chain 
 """

from dataclasses import dataclass, field
from typing import List


@dataclass
class SupplyChainTopography:
    number_of_stages: int = field(default=4)
