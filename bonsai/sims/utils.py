from ast import Raise
from datetime import datetime
import numpy as np
from random import randint as randint
from typing import List

def assign_env_config(self, kwargs):
    for key, value in kwargs.items():
        setattr(self, key, value)
    if hasattr(self, 'env_config'):
        for key, value in self.env_config.items():
            # Check types based on default settings
            if hasattr(self, key):
                if type(getattr(self, key)) == np.ndarray:
                    setattr(self, key, value)
                else:
                    setattr(self, key,
                            type(getattr(self, key))(value))
            else:
                raise AttributeError(f"{self} has no attribute, {key}")

# Get Ray to work with gym registry


def create_env(config, *args, **kwargs):
    if type(config) == dict:
        env_name = config['env']
    else:
        env_name = config
    if env_name == 'Knapsack-v0':
        from or_gym.envs.classic_or.knapsack import KnapsackEnv as env
    elif env_name == 'Knapsack-v1':
        from or_gym.envs.classic_or.knapsack import BinaryKnapsackEnv as env
    elif env_name == 'Knapsack-v2':
        from or_gym.envs.classic_or.knapsack import BoundedKnapsackEnv as env
    elif env_name == 'Knapsack-v3':
        from or_gym.envs.classic_or.knapsack import OnlineKnapsackEnv as env
    elif env_name == 'BinPacking-v0':
        from or_gym.envs.classic_or.binpacking import BinPackingEnv as env
    elif env_name == 'BinPacking-v1':
        from or_gym.envs.classic_or.binpacking import BinPackingLW1 as env
    elif env_name == 'BinPacking-v2':
        from or_gym.envs.classic_or.binpacking import BinPackingPP0 as env
    elif env_name == 'BinPacking-v3':
        from or_gym.envs.classic_or.binpacking import BinPackingPP1 as env
    elif env_name == 'BinPacking-v4':
        from or_gym.envs.classic_or.binpacking import BinPackingBW0 as env
    elif env_name == 'BinPacking-v5':
        from or_gym.envs.classic_or.binpacking import BinPackingBW1 as env
    elif env_name == 'VMPacking-v0':
        from or_gym.envs.classic_or.vmpacking import VMPackingEnv as env
    elif env_name == 'VMPacking-v1':
        from or_gym.envs.classic_or.vmpacking import TempVMPackingEnv as env
    elif env_name == 'PortfolioOpt-v0':
        from or_gym.envs.finance.portfolio_opt import PortfolioOptEnv as env
    elif env_name == 'TSP-v0':
        raise NotImplementedError('{} not yet implemented.'.format(env_name))
    elif env_name == 'VehicleRouting-v0':
        from or_gym.envs.classic_or.vehicle_routing import VehicleRoutingEnv as env
    elif env_name == 'VehicleRouting-v1':
        from or_gym.envs.classic_or.vehicle_routing import VehicleRoutingEnv as env
    elif env_name == 'NewsVendor-v0':
        from or_gym.envs.classic_or.newsvendor import NewsvendorEnv as env
    elif env_name == 'InvManagement-v0':
        from or_gym.envs.supply_chain.inventory_management import InvManagementBacklogEnv as env
    elif env_name == 'InvManagement-v1':
        from or_gym.envs.supply_chain.inventory_management import InvManagementLostSalesEnv as env
    elif env_name == 'InvManagement-v2':
        from or_gym.envs.supply_chain.network_management import NetInvMgmtBacklogEnv as env
    elif env_name == 'InvManagement-v3':
        from or_gym.envs.supply_chain.network_management import NetInvMgmtLostSalesEnv as env
    else:
        raise NotImplementedError(
            'Environment {} not recognized.'.format(env_name))
    return env

def calc_runtime(f):
    def wrap(*args, **kwargs):
        s_time = datetime.now()
        x = f(*args, **kwargs)
        print('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>')
        print(f"time taken: {datetime.now() - s_time}")
        return x
    return wrap

def make_lead_profile(profile: int = 2, periods: int = 360, n_stages: int =3, min_lead: List[int] = [2, 4, 6], max_lead: List[int]=[6, 8, 10]) -> List[List[int]]:
    '''
    Makes lead profile according to profile type and the length of the episode.
    2: Step randomized, i.e lead changing every 5 days. The min and max is set to be 3 and 20. 
    3: Step and frequency randomized, i.e lead chaning every 3- 10 days 
    4: lead time changing randomly 
    '''
    lead_all = []
    init_lead = list(np.random.randint(2, 5, n_stages))
    for i in range(0, n_stages):
        lead_all.append(periods*[init_lead[i]])
    if profile == 1:
        print("Lead profile: fixed 2,3,4")
        for i in range(0, periods):
            for j in range(n_stages):
                lead_all[j][i] = j+2
    if profile == 2:
        print("Lead profile: fixed 3,4,5")
        for i in range(0, periods):
            for j in range(n_stages):
                lead_all[j][i] = j+3
    if profile == 3: 
        print("Lead profile: fixed 2,4,6")  
        for i in range(0, periods):
            for j in range(n_stages):
                lead_all[j][i] = 2*j+2 
    if profile == 4: 
        print("Lead profile: fixed 2,3,6")  
        for i in range(0, periods):
            for j in range(n_stages):
                lead_all[j][i] = 2*j+2  
    if profile == 5: 
        print("Lead profile: fixed 2,3,7")  
        for i in range(0, periods):
            for j in range(n_stages):
                lead_all[j][i] = 2*j+2    
    if profile == 6:  
        print("Lead profile: fixed 3,4,6") 
        for i in range(0, periods):
            for j in range(n_stages):
                lead_all[j][i] = int(1.5*j)+3        
    if profile == 7:
        print('Lead profile: constant changes across levels every 20 iteration')
        for j in range(0, n_stages):
            if i%20==0:
                lead_all[j][i] = lead_all[j][i-1] + randint(-1, 2)
            else:
                lead_all[j][i] = lead_all[j][i-1]

    if profile == 8:
        print('Lead profile: changes up to five days every 20 days')
        for i in range(1, periods):
            delta = np.array(n_stages * [0])
            if i % 20 == 0:
                delta = np.random.randint(-1, 1, n_stages)
            else:
                delta = np.array(n_stages * [0])
            for j in range(0, n_stages):
                lead_all[j][i] = lead_all[j][i-1] + delta[j]
    if profile == 9:
        print('Lead profile: changes upt to five days at random 10-30 days')
        for i in range(1, periods):
            delta = n_stages * [0]
            rand_freq = np.random.randint(10, 30)
            if i % rand_freq == 0:
                delta = np.random.randint(-3, 3, n_stages)
            else: 
                delta = np.array(n_stages * [0])
            for j in range(0, n_stages):
                lead_all[j][i] = lead_all[j][i-1] + delta[j]

    if profile >= 10:
        ValueError(f'profile>= {profile} is not implemented yet')

    # impose upper and lower bound
    for i in range(0, n_stages):
        for j in range(0, periods):
            lead_all[i][j] = min(max(lead_all[i][j], min_lead[i]), max_lead[i])
          
    return lead_all



