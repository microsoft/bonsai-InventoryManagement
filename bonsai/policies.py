"""
Fixed policies to test our sim integration with. These are intended to take
Brain states and return Brain actions.
"""

from ctypes import Union
import random
from typing import Dict, List, Any
import requests
import numpy as np
import json
from typing import Optional


def random_policy(state: Dict[str, float]):
    """
    Ignore the state, move randomly.
    """
    action = {
        "order_stage0": random.randint(15, 25),
        "order_stage1": random.randint(15, 25),
        "order_stage2": random.randint(15, 25)
    }
    return action


def zero_policy(state: Dict[str, float]):
    """
    Ignore the state, move randomly.
    """
    action = {
        "safety_stock_stage0": 0,
        "safety_stock_stage1": 0,
        "safety_stock_stage2": 0
    }
    return action


def random_safety_policy(state: Dict[str, float]):
    """
    Ignore the state, move randomly.
    """
    action = {
        "safety_stock_stage0": np.random.randint(0, 10),
        "safety_stock_stage1": np.random.randint(0, 10),
        "safety_stock_stage2": np.random.randint(0, 10)
    }
    return action


def heuristic_policy(state: Dict[str, float]):
    """
    Ignore the state, move randomly.
    """
    action = {
        "order_stage0": state["demand_forecast"][2],
        "order_stage1": state["demand_forecast"][5],
        "order_stage2": state["demand_forecast"][8]
    }
    return action


def brain_policy_old(
    state: Dict[str, float], exported_brain_url: str = "http://localhost:5000"
):

    prediction_endpoint = f"{exported_brain_url}/v1/prediction"
    response = requests.get(prediction_endpoint, json=state)

    return response.json()


def brain_policy(state: Dict[str, Any], exported_brain_url: str = "http://localhost:5000"
                 ):
    predictionPath = "/v1/prediction"
    headers = {
        "Content-Type": "application/json"
    }
    endpoint = exported_brain_url + predictionPath

    requestBody = {

        'transit_orders': [int(i) for i in state['transit_orders']],
        'demand_actual': int(state['demand_actual']),
        'demand_forecast': [int(i) for i in state['demand_forecast']],
        'demand_sigma': [int(i) for i in state['demand_sigma']],
        'inventory': [int(i) for i in state['inventory']],
        'leads': [int(i) for i in state['leads']],
        'missed_sale_to_inventory_cost_ratio': int(state['missed_sale_to_inventory_cost_ratio'])
    }
    print('request body:', requestBody)

    response = requests.post(
        endpoint,
        data=json.dumps(requestBody),
        headers=headers
    )
    return response.json()


def forget_memory(
    url: str = "http://localhost:5000/v1"
):
    # Reset the Memory vector because exported brains don't understand episodes
    response = requests.delete(url)
    if response.status_code == 204:
        print('Resetting Memory vector in exported brain...')
    else:
        print('Error: {}'.format(response.status_code))


def or_policy(state: Dict[str, float]):
    demand_forecast = state['demand_forecast']
    action = {
        "order_stage0": demand_forecast[1],
        "order_stage1": demand_forecast[5],
        "order_stage2": demand_forecast[9],
    }
    return action


def safety_policy(state: Dict[str, float]):
    demand_sigma = state['demand_sigma']
    action = {
        "safety_stock_stage0": int(1 + np.sqrt(np.mean(demand_sigma[2:5]))),
        "safety_stock_stage1": int(1 + np.sqrt(np.mean(demand_sigma[3:7]))),
        "safety_stock_stage2": int(1 + np.sqrt(np.mean(demand_sigma[4:10]))),
    }
    return action
