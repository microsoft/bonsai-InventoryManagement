'''
__author__: Hossein Khadivi Heris
@Microsoft 06/19/2022
'''
from dataclasses import dataclass, field, asdict, replace
from abc import ABC, abstractmethod
from fileinput import filename 
import json 
import random
import string
import numpy as np
from typing import Union, List, Deque, Dict, Type
from collections import deque
from numpy.random import randint
from .constraints import MultiSKUConstraints
from .chain_definition import SupplyChainTopology
from .utils import make_lead_profile
from .utils_demand import gen_custom_demand

'''
Follows a factory design pattern to create large number of skus with defined properties. 
'''

def generate_name() -> str:
    return "".join(random.choices(string.ascii_uppercase, k=12))


def load_sku_info(filename: str):
    with open(filename) as f:
        info = json.load(f)
    return info


def generate_random_sku_info_as_json(n_skus = 1, n_stages = 4, ratio = 100): 
    missed_sale_to_inventory_cost_ratio = ratio
    sku_info = {}
    for i in range(0,n_skus):
        vol = np.round(1 + i*0.1,2)
        unit_price = np.round([(1 + i*0.2)*(n_stages - j)/n_stages for j in range(n_stages)],2)
        missed_sale_cost = np.round(unit_price/4,2) # 1/4 of price is the profit for each sku, as such missed sale 
        unit_cost = np.round(unit_price-missed_sale_cost,2)
        missed_sale_cost = np.round(unit_price/10,2) 
        storage_cost = np.round(missed_sale_cost/missed_sale_to_inventory_cost_ratio,5)
        sku_info[i] = asdict(SKUGenericInfo(id = i, volume = vol,
                           storage_cost = storage_cost.tolist(), 
                           unit_price = unit_price.tolist(),
                           unit_cost = unit_cost.tolist(),
                           batch = n_stages*[0], 
                           missed_sale_cost=missed_sale_cost.tolist()))
        
    with open('sku_properties.json', "w") as f:
        json.dump(sku_info, f)
    return sku_info

def generate_random_sku_info_as_json2(n_skus = 1, n_stages = 4): 
    sku_info = {}
    missed_sale_to_inventory_cost_ratio = np.power(10, np.random.uniform(0,3, n_skus))
    for i in range(0,n_skus):
        vol = np.round(1 + i*0.1,2)
        random_price = np.random.uniform(1,100)
        unit_price = np.round([random_price*(n_stages - j)/n_stages for j in range(n_stages)],2)
        missed_sale_cost = np.round(unit_price/4,2) # 1/4 of price is the profit for each sku, as such missed sale 
        unit_cost = np.round(unit_price-missed_sale_cost,2)
        storage_cost = np.round(missed_sale_cost/int(missed_sale_to_inventory_cost_ratio[i]),5)
        sku_info[i] = asdict(SKUGenericInfo(id = i, volume = vol,
                           storage_cost = storage_cost.tolist(), 
                           unit_price = unit_price.tolist(),
                           unit_cost = unit_cost.tolist(),
                           batch = n_stages*[0], 
                           missed_sale_cost=missed_sale_cost.tolist()))   
    with open('sku_properties.json', "w") as f:
        json.dump(sku_info, f)
    return sku_info


@dataclass
class demand_info:
    demand_actual: List[int] 
    demand_forecast: List[int]
    forecast_sigma: List[int] 


# any property of sku should be in the following data class 
@dataclass
class SKUGenericInfo:
    '''
    General sku information, mostly static but may change over time as well. 
    '''
    id: int 
    volume: float
    storage_cost: List[float] #at each stage  
    missed_sale_cost: List[float] # selling price - replenishment cost 
    unit_price: List[float] # selling price at each stage  
    unit_cost: List[float] # replenishment cost at each stage  
    batch: List[int]
    name: str = field(init = True, default_factory=generate_name)

                    
@dataclass
class SKUDynamicInfo:
    '''
    SKU demand and order information. All change dynamically over episode and iterations 
    id is common between static and dynamic information 
    Note that transit orders reflect orders for a single sku across all levels of the supplychain
    '''
    id: int
    demand_info: demand_info # loosly defined here. reads data of type demand_info 
    transit_order: List[Deque]   # [[level 0_deque], [level_1_deque], ...]
    inventory: List[int]   # [[level_0_inv], [level_1_inv], ...]
    safety: List[int]            # [level0, ... level n]
    leads: List[int]   # current lead time 
    lead_profile: List[List[int]]  # [[level0_lead_sequence_time], [level1_lead_sequence_time, ...]
    period: int = field(init = True, default = 0)
    forecast_window : int  = field(init = True, default = 10)
    @property
    def iteration_demand_forecast(self) -> List[int]:
        return self.demand_info.demand_forecast[self.period+1: self.period+self.forecast_window+1].tolist()
    @property
    def iteration_forecast_sigma(self) -> List[int]:
        return self.demand_info.forecast_sigma[self.period+1: self.period+self.forecast_window+1].tolist()    
    @property
    def iteration_actual_demand(self) -> int:
        return self.demand_info.demand_actual[self.period]
    
@dataclass
class SKUsCollection:
    '''
    dictionary that maps, sku id to sku information 
    '''
    generic: Dict[int, SKUGenericInfo]
    dynamic: Dict[int, SKUDynamicInfo]
      
    
class InfoFactory(ABC):
    @abstractmethod
    def _prepare_generic_info(self):
        """
        Mostly static info such as pricing, etc
        """
    @abstractmethod    
    def _prepare_dynamic_info(self):
        """
        Mostly dynamic info suc as transit orders, etc 
        """
    @abstractmethod
    def _prepare_constraints(self):
        """constraints applied to multi sku such as inventory capacity 
        """
@dataclass()
class SKUInfoFactory(InfoFactory):

    constraints: MultiSKUConstraints
    sku_count: int = 10
    n_levels: int = 3
    info = SKUsCollection(generic={}, dynamic ={})
    def __init__(self, sku_count: int = 10, topology: SupplyChainTopology = SupplyChainTopology(), config: dict = {}):
        self.sku_count = sku_count
        self.n_levels = (topology.number_of_stages-1) # last stage has infinite supply for dynamic property 
        try:
            cost_ratio = config["missed_sale_to_inventory_cost_ratio"]
        except:
            cost_ratio = 100 # default 
        generate_random_sku_info_as_json(n_skus = sku_count, n_stages = topology.number_of_stages, ratio = cost_ratio)
        self.sku_info = load_sku_info(filename = 'sku_properties.json')
        self._prepare_generic_info()
        self._prepare_dynamic_info(n_levels=self.n_levels, config = config)
        self._prepare_constraints(config = config)
    
    def _prepare_constraints(self, config: dict = {}):
        try:
            inv_cap = self.n_levels*[int(config["inventory_capacity"])]
        except: 
            inv_cap = self.n_levels*[self.sku_count*25]
        

        self.constraints = MultiSKUConstraints(inventory_capacity = inv_cap,
                                               supply_capacity= [(100-10*i) for i in range(self.n_levels)])
    
    def update_iteration_for_all_skus(self, iteration: int = 0):
        '''
        takes the iteration (period) number and updates demand information for all the skus
        '''
        for s in range(self.sku_count):
            self.info.dynamic[s].period = iteration
    
    def update_transit_orders_for_all_skus(self, iteration_purchase_orders: List[List[int]]):
        '''
        receives action as 2d list [[sku0_all_levels],[sku1_all_levels], ...]
        '''
        for s in range(self.sku_count):
            for j in range(self.n_levels):
                self.info.dynamic[s].transit_order[j].appendleft(int(iteration_purchase_orders[s][j]))
    
    def update_lead_for_all_skus(self):
        for s in range(self.sku_count):
            self.info.dynamic[s].leads = []
            for j in range(self.n_levels):
                print(f"sku{s} level{j} has lead {self.info.dynamic[s].lead_profile[j][self.info.dynamic[s].period]}")
                self.info.dynamic[s].leads.append(int(self.info.dynamic[s].lead_profile[j][self.info.dynamic[s].period]))
    
    def update_inventory_for_all_skus(self, inventory_levels: List[List[int]]):
        for s in range(self.sku_count):
            self.info.dynamic[s].inventory = self.info.dynamic[s].inventory = inventory_levels[s]                           
        
    def generate_multi_sku_info_from_json(self, file_name = 'sku_properties.json'):
        try:
            self.sku_info = load_sku_info(filename = file_name)
        except:
            raise Exception("didn't find json file, make sure sku_properties.json exists ..")
    def _prepare_generic_info(self):   
        for i in range(self.sku_count):
            info = self.sku_info[str(i)]
            print(info)
            self.info.generic[i] = (replace(SKUGenericInfo(id = i, volume = info['volume'], 
                                            storage_cost= info['storage_cost'],
                                            unit_price=info['unit_price'], 
                                            unit_cost = info['unit_cost'],
                                            batch = 0, 
                                            missed_sale_cost= info['missed_sale_cost']), name= info['name']))
    def _prepare_dynamic_info(self, sigmax: int = 7,
                offset: float = 20, amp1: float = 5,
                ph1: float = 0.02, amp2: float = 3, 
                ph2: float = 0.05, randvar: float = 1,
                n_levels = 3, config = None):


        try:
            sigmax = config['sigmax']
        except:
            pass        
        
        try:
            offset = config['offset']
        except:
            pass
    
        try:
            amp1 = config['amp1']
        except:
            pass
        
        try:
            ph1 = config['ph1']
        except:
            pass
        
        try:
            amp2 = config['amp2']
        except:
            pass
    
        try: 
            ph2 = config['ph2']
        except:
            pass
        
        try:
            randvar = config ['randvar']
        except:
            pass
        
        try:
            n_levels = config['n_levels']
        except:
            pass
        
        try:
            config_profile = config["lead_profile"]
        except: 
            config_profile = 1 
            
        print(offset)
        init_transit_order_min = 0
        init_transit_order_max = 1
        assert(init_transit_order_min<init_transit_order_max)
        init_inventory_min = 5
        init_inventory_max = 10
        init_safety_stock = n_levels*[2] 
        for i in range(self.sku_count):
            episode_length = 356
            days_offset = 50
            arbitrary_start_day = np.random.randint(
                0, episode_length-days_offset)
            arbitrary_end_day = arbitrary_start_day + episode_length
            synthetic_demand = gen_custom_demand(sigmax = sigmax,
                                        offset = offset, 
                                        amp1 = amp1,
                                        ph1 = ph1,
                                        amp2 = amp2,
                                        ph2 = ph2,
                                        randvar = randvar,
                                        start_index= arbitrary_start_day,
                                        end_index = arbitrary_end_day)
            self.info.dynamic[i]= SKUDynamicInfo(id = i,
                                                 demand_info= demand_info(demand_actual=synthetic_demand["demand_actual"],
                                                              demand_forecast = synthetic_demand["demand_forecast"],
                                                              forecast_sigma=synthetic_demand["forecast_sigma"]) ,
                                                        transit_order = [deque(randint(init_transit_order_min, init_transit_order_max, 45), maxlen=45)
                                                                for j in range(0, n_levels)],
                                                        inventory = list(randint(init_inventory_min, init_inventory_max, n_levels)),
                                                        safety = init_safety_stock,
                                                        leads = [2,3,4], #[randint(i*2,(i+1)*2) for i in range(1,n_levels+1)], # [2,3,4] or list(range(2, n_levels+2))
                                                        lead_profile = make_lead_profile(profile = config_profile,
                                                            n_stages = self.n_levels,
                                                            min_lead =[i + 2 for i in range(self.n_levels)], 
                                                            max_lead = [self.n_levels + 2 + i for i in range(self.n_levels)]
                                                        )
                                                        )   
            
              
@dataclass()
class SKUInfoFactoryRandom(InfoFactory):
    '''
    Factory with
    lead: selected randomly from available lead times 1-9 
    cost_ratio: randomly selected (1-1000)
    storage cost: randomly created 
    missed_sale_cost: created using cost_ratio and the storage cost 
    '''
    constraints: MultiSKUConstraints
    sku_count: int = 10
    n_levels: int = 3
    info = SKUsCollection(generic={}, dynamic ={})
    def __init__(self, sku_count: int = 10, topology: SupplyChainTopology = SupplyChainTopology(), config: dict = {}):
        self.sku_count = sku_count
        self.n_levels = (topology.number_of_stages-1) # last stage has infinite supply for dynamic property 
        try:
            cost_ratio = config["missed_sale_to_inventory_cost_ratio"]
        except:
            cost_ratio = 100 # default 
        generate_random_sku_info_as_json2(n_skus = sku_count, n_stages = topology.number_of_stages)
        self.sku_info = load_sku_info(filename = 'sku_properties.json')
        self._prepare_generic_info()
        self._prepare_dynamic_info(n_levels=self.n_levels, config = config)
        self._prepare_constraints(config = config)
    
    def _prepare_constraints(self, config: dict = {}):
        try:
            inv_cap = self.n_levels*[int(config["inventory_capacity"])]
        except: 
            inv_cap = self.n_levels*[self.sku_count*25]
        

        self.constraints = MultiSKUConstraints(inventory_capacity = inv_cap,
                                               supply_capacity= [(100-10*i) for i in range(self.n_levels)])
    
    def update_iteration_for_all_skus(self, iteration: int = 0):
        '''
        takes the iteration (period) number and updates demand information for all the skus
        '''
        for s in range(self.sku_count):
            self.info.dynamic[s].period = iteration
    
    def update_transit_orders_for_all_skus(self, iteration_purchase_orders: List[List[int]]):
        '''
        receives action as 2d list [[sku0_all_levels],[sku1_all_levels], ...]
        '''
        for s in range(self.sku_count):
            for j in range(self.n_levels):
                self.info.dynamic[s].transit_order[j].appendleft(int(iteration_purchase_orders[s][j]))
    
    def update_lead_for_all_skus(self):
        for s in range(self.sku_count):
            self.info.dynamic[s].leads = []
            for j in range(self.n_levels):
                print(f"sku{s} level{j} has lead {self.info.dynamic[s].lead_profile[j][self.info.dynamic[s].period]}")
                self.info.dynamic[s].leads.append(int(self.info.dynamic[s].lead_profile[j][self.info.dynamic[s].period]))
    
    def update_inventory_for_all_skus(self, inventory_levels: List[List[int]]):
        for s in range(self.sku_count):
            self.info.dynamic[s].inventory = self.info.dynamic[s].inventory = inventory_levels[s]                           
        
    def generate_multi_sku_info_from_json(self, file_name = 'sku_properties.json'):
        try:
            self.sku_info = load_sku_info(filename = file_name)
        except:
            raise Exception("didn't find json file, make sure sku_properties.json exists ..")
    def _prepare_generic_info(self):   
        for i in range(self.sku_count):
            info = self.sku_info[str(i)]
            print(info)
            self.info.generic[i] = (replace(SKUGenericInfo(id = i, volume = info['volume'], 
                                            storage_cost= info['storage_cost'],
                                            unit_price=info['unit_price'], 
                                            unit_cost = info['unit_cost'],
                                            batch = 0, 
                                            missed_sale_cost= info['missed_sale_cost']), name= info['name']))
    def _prepare_dynamic_info(self, sigmax: int = 7,
                offset: float = 20, amp1: float = 5,
                ph1: float = 0.02, amp2: float = 3, 
                ph2: float = 0.05, randvar: float = 1,
                n_levels = 3, config = None):

        try:
            sigmax = config['sigmax']
        except:
            pass        
        
        try:
            offset = config['offset']
        except:
            pass
    
        try:
            amp1 = config['amp1']
        except:
            pass
        
        try:
            ph1 = config['ph1']
        except:
            pass
        
        try:
            amp2 = config['amp2']
        except:
            pass
    
        try: 
            ph2 = config['ph2']
        except:
            pass
        
        try:
            randvar = config ['randvar']
        except:
            pass
        
        try:
            n_levels = config['n_levels']
        except:
            pass
        
        # try:
        #     config_profile = config["lead_profile"]
        # except: 
        #     config_profile = 1 
          
            
        print(offset)
        init_transit_order_min = 0
        init_transit_order_max = 1
        assert(init_transit_order_min<init_transit_order_max)
        # init_transit_orders = [deque(randint(init_transit_order_min, init_transit_order_max, 45), maxlen=45)
        #                     for j in range(0, n_levels)]
        init_inventory_min = 5
        init_inventory_max = 10
        init_safety_stock = n_levels*[2] 
        for i in range(self.sku_count):
            episode_length = 356
            days_offset = 50
            arbitrary_start_day = np.random.randint(
                0, episode_length-days_offset)
            arbitrary_end_day = arbitrary_start_day + episode_length
            synthetic_demand = gen_custom_demand(sigmax = sigmax,
                                        offset = offset, 
                                        amp1 = amp1,
                                        ph1 = ph1,
                                        amp2 = amp2,
                                        ph2 = ph2,
                                        randvar = randvar,
                                        start_index= arbitrary_start_day,
                                        end_index = arbitrary_end_day)
            self.info.dynamic[i]= SKUDynamicInfo(id = i,
                                                 demand_info= demand_info(demand_actual=synthetic_demand["demand_actual"],
                                                              demand_forecast = synthetic_demand["demand_forecast"],
                                                              forecast_sigma=synthetic_demand["forecast_sigma"]) ,
                                                        transit_order = [deque(randint(init_transit_order_min, init_transit_order_max, 45), maxlen=45)
                                                                for j in range(0, n_levels)],
                                                        inventory = list(randint(init_inventory_min, init_inventory_max, n_levels)),
                                                        safety = init_safety_stock,
                                                        leads = [2,3,4], #[randint(i*2,(i+1)*2) for i in range(1,n_levels+1)], # [2,3,4] or list(range(2, n_levels+2))
                                                        lead_profile = make_lead_profile(profile = np.random.randint(0,9),
                                                            n_stages = self.n_levels,
                                                            min_lead =[i + 2 for i in range(self.n_levels)], 
                                                            max_lead = [self.n_levels + 2 + i for i in range(self.n_levels)]
                                                        )
                                                        )   
            
                                                           
if __name__ == "__main__":
    #generate_random_sku_info_as_json(n_skus = 10)
    skus = SKUInfoFactoryRandom(sku_count=10)
    print("------------------------")
    print(skus.sku_count)
    print("-------------------------")
    print(skus.info.generic[0])
    print(skus.info.dynamic[1])
    skus.info.dynamic[1].inventory =[-15,-5,-5]
    skus.info.generic[9].unit_price = 10
    print('price is:', skus.info.generic[9].unit_price)
    print('num sku:',skus.sku_count)
    print('get attr inv:', skus.info.dynamic[1].inventory)
    print(skus.info.dynamic[1].demand_info.demand_actual)
    #skus.constraints.inventory_capacity = [600, 800, 200]
    print('inventory_capacity:',skus.constraints.inventory_capacity)
    # # generate_random_sku_info_as_json(n_skus=10)
    print(skus.info.dynamic[8])
    print(skus.info.dynamic[1].inventory)
    
    
    