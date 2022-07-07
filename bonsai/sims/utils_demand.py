from typing import List
#from anyio import start_blocking_portal
import numpy as np
from random import random
import numpy as np
from matplotlib import pyplot as plt


def gen_custom_demand(sigmax: int = 6, offset: float = 5, amp1: float = 5, ph1: float = 0.02, amp2: float = 3, ph2: float = 0.05, randvar: float = 15,
                      start_index = 0, end_index = float('Inf')):
    """
    To generate synthetic forecast data that has
    (1) peaks and valleys using two sinusoidal function 
    (2) randomness added to forecasts for un-smooth and real world-like data

    Actual demand data by adding normal noise with 
    (1) known but variable sigma 
    (2) mean = 0 
    """
    n = 365*3  # n days in a year
    x = np.array([i for i in range(1, n+1)])
    var1 = np.array([np.round(np.random.randint(0, randvar))
                    for i in range(1, n+1)])
    demand_forecast = offset + amp1*np.sin(ph1*x) + amp2*np.sin(ph2*x) + var1
    # sigma for each forecasted value
    if sigmax == 0:
        demand_actual = demand_forecast
        sigma = np.array([0 for i in range(1, n+1)])
    else:
        sigma = np.array([np.round(np.random.randint(0, sigmax))
                         for i in range(1, n+1)])
        demand_actual = demand_forecast + \
            np.array([np.random.normal(0, sig) for sig in sigma])

    demand_forecast = np.maximum(demand_forecast, 0)
    demand_actual = np.maximum(demand_actual, 0)
    
    if end_index == float('Inf'):
        demand_info = {
            'demand_actual': demand_actual[start_index:].astype(int),
            'demand_forecast': demand_forecast[start_index:].astype(int),
            'forecast_sigma': sigma[start_index].astype(int)
        }
    else:
        demand_info = {
            'demand_actual': demand_actual[start_index:end_index].astype(int),
            'demand_forecast': demand_forecast[start_index:end_index].astype(int),
            'forecast_sigma': sigma[start_index:end_index].astype(int)
        }
    return demand_info


def gen_assess_demand(assess_profile=6):
    '''
    Custom demand profile that are simplistic and are meant to check the explainability of the
    brain performance 
    profile 6: 
        demand = fixed value (20), sudden jump for 1 iteration (30) and then return back to the previous fix level (20) 
    profile 7:
        demand = fixed value (20), suddent jump (30) for 20 iterations and then return back to the previous level (20) 
    profile 8:
        demand = fixed value (20), ramp up for 5 iteration(20-->30), then stay at high value (30) for 10 iterations, ramp down (30-->20) for 5 iterations, and return to previous level
    '''
    n = 365*3  # n days in a year
    x = np.array([1 for i in range(1, n+1)])
    # Sigma always = 0 as forecasts are accurate
    sigma = np.array([0 for i in range(1, n+1)])

    if assess_profile == 6:
        y = np.array([1 if i == 20 else 0 for i in range(1, n+1)])
        demand_forecast = x*y*10 + 25
    elif assess_profile == 7:
        y = np.array([0 if i < 20 or i > 40 else 1 for i in range(1, n+1)])
        demand_forecast = x*y*10 + 25
    elif assess_profile == 8:
        y = []
        for i in range(1, n+1):
            if i <= 20 or i >= 40:
                y.append(0)
            elif i > 20 and i <= 25:
                y.append((i-20)/5)
            elif i > 35 and i < 40:
                y.append((40-i)/5)
            else:
                y.append(1)
        demand_forecast = x*np.array(y)*10+25
    elif assess_profile == 9:
        y = np.array([-1 if i == 20 else 0 for i in range(1, n+1)])
        demand_forecast = x*y*10 + 25
    elif assess_profile == 10:
        y = np.array([0 if i < 20 or i > 40 else -1 for i in range(1, n+1)])
        demand_forecast = x*y*10 + 25
    elif assess_profile == 11:
        y = []
        for i in range(1, n+1):
            if i <= 20 or i >= 40:
                y.append(0)
            elif i > 20 and i <= 25:
                y.append(-(i-20)/5)
            elif i > 35 and i < 40:
                y.append(-(40-i)/5)
            else:
                y.append(-1)
        demand_forecast = x*np.array(y)*10+25
    elif assess_profile == 12:
        y = []
        for i in range(1, n+1):
            if i <= 10:
                y.append(0)
            else:
                y.append(25)
        demand_forecast = np.array(y)
    else:
        raise("profile not supported")

    # sigma for each forecasted value

    demand_actual = demand_forecast

    demand_info = {
        'demand_actual': demand_actual.astype(int),
        'demand_forecast': demand_forecast.astype(int),
        'forecast_sigma': sigma.astype(int)
    }
    return demand_info


if __name__ == "__main__":
    import random
    import or_gym

    env_name = 'InvManagement-v1'
    demand_info = gen_custom_demand(
        sigmax=5, offset=20, amp1=5, ph1=0.02, amp2=3, ph2=0.05)
    #demand_info = gen_custom_demand(sigmax=1,offset=0,amp1=0,ph1=0.00,amp2=0,ph2=0.00, randvar=1)
    episode_period = 365

    demand_actual = demand_info["demand_actual"]
    demand_forecast = demand_info["demand_forecast"]
    demand_sigma = demand_info["forecast_sigma"]
    print(demand_actual)
    print(demand_forecast)
    print(demand_sigma)

    plt.plot(demand_forecast, 'r')
    plt.plot(demand_actual, 'b')
    plt.show()
    arbitrary_start_day = np.random.randint(0, 356*2-1-episode_period)
    demand_actual = demand_info["demand_actual"][arbitrary_start_day:
                                                 arbitrary_start_day + episode_period]
    demand_forecast = demand_info["demand_forecast"][arbitrary_start_day:
                                                     arbitrary_start_day + episode_period]
    demand_sigma = demand_info["forecast_sigma"][arbitrary_start_day:
                                                 arbitrary_start_day + episode_period]
    plt.plot(demand_forecast, 'r')
    plt.plot(demand_actual, 'b')
    plt.show()
    env_config = {'dist': 5, 'user_D': demand_actual,
                  'periods': episode_period}
    env = or_gym.make(env_name, env_config=env_config)

    eps = 1000
    rewards = []
    for i in range(eps):
        env.reset()
        reward = 0
        for j in range(100):
            #action = [random.randint(50, 500),random.randint(50, 500),random.randint(50, 500)]
            # action = env.action_space.sample()
            action = [100, 50, 50]
            s, r, done, _ = env.step(action)
            reward += r
            print(reward)
            if done:
                rewards.append(reward)
                break
