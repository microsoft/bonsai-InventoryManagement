
#!/usr/bin/env python3

"""
MSFT Bonsai SDK3 Template for Simulator Integration using Python
Copyright 2021 Microsoft
Usage:
  For registering simulator with the Bonsai service for training:
    python main.py \
           --workspace <workspace_id> \
           --accesskey="<access_key> \
  Then connect your registered simulator to a Brain via UI
  Alternatively, one can set the SIM_ACCESS_KEY and SIM_WORKSPACE as
  environment variables.
"""

from asyncio import futures
from functools import partial
import json
import random
import math
import time
import os
import pathlib
from dotenv import load_dotenv, set_key
import datetime
from collections import deque
from typing import Dict, Any, Union
from microsoft_bonsai_api.simulator.client import BonsaiClientConfig, BonsaiClient
from microsoft_bonsai_api.simulator.generated.models import (
    SimulatorState,
    SimulatorInterface,
    SimulatorSessionResponse,
)
from azure.core.exceptions import HttpResponseError
import argparse
import numpy as np
from send2trash import TrashPermissionError
from policies import random_policy, brain_policy, forget_memory, safety_policy, zero_policy, random_safety_policy
import pdb
from sims.supply_chain.multi_sku import SKUInfoFactoryRandom
from sims.supply_chain.chain_definition import SupplyChainTopology
from sims.supply_chain.inventory_management import InvManagementLostSalesMultiSKUEnv
from sims.supply_chain.mip_solver import MipSolver


def make_multi_sku_env(config: dict = {'number_of_stages': 4, 'number_of_sku': 1}) -> InvManagementLostSalesMultiSKUEnv:
    '''
    Input: number of skus and stages in multi-echelon inventory management 
    Output: multisku simulation environment 
    Note: skus are built using sku info factory
    '''
    my_chain = SupplyChainTopology(
        number_of_stages=config["number_of_stages"])
    skus = SKUInfoFactoryRandom(
        sku_count=config["number_of_sku"], topology=my_chain, config=config)
    print(skus.info)
    # time.sleep(10)
    return InvManagementLostSalesMultiSKUEnv(skus)


LOG_PATH = "logs"


def ensure_log_dir(log_full_path):
    """
    Ensure the directory for logs exists — create if needed.
    """
    print(f"logfile: {log_full_path}")
    logs_directory = pathlib.Path(log_full_path).parent.absolute()
    print(f"Checking {logs_directory}")
    if not pathlib.Path(logs_directory).exists():
        print(
            "Directory does not exist at {0}, creating now...".format(
                str(logs_directory)
            )
        )
        logs_directory.mkdir(parents=True, exist_ok=True)


class TemplateSimulatorSession:
    def __init__(
        self,
        render: bool = False,
        log_data: bool = False,
        log_file_name: str = None,
        env_name: str = "IMS",
    ):
        # Initialize python api for simulator
        self.simulator = make_multi_sku_env()
        self.env_name = env_name
        #self.state = self.simulator.reset().tolist()
        self.sim_reward = 0
        self.iter_count = 0
        self.sum_cost_action_freq = 0
        self.constraint_relaxation = 0
        self.mip_cost_config = "Safety"
        self.missed_sale_to_inventory_cost_ratio = 10
        self.sim_terminal = False
        self.render = render
        self.log_data = log_data
        self.config = None
        if not log_file_name:
            current_time = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
            log_file_name = current_time + "_" + env_name + "_log.csv"

        self.log_full_path = os.path.join(LOG_PATH, log_file_name)
        ensure_log_dir(self.log_full_path)

    def get_state(self) -> Dict[str, Any]:
        """Called to retreive the current state of the simulator. """

        transit_orders = []
        for level in range(self.simulator.skus.n_levels):
            transit_orders.append(
                list(self.simulator.skus.info.dynamic[0].transit_order[level])[0:15])
        transit_orders = [
            item for sublist in transit_orders for item in sublist]
        # make sure this matches that of platform
        self.simulator.skus.info.dynamic[0].forecast_window = 15
        sim_state = {}
        sim_state['transit_orders'] = [int(order) for order in transit_orders]
        # ["demand_actual"][sim.simulator.period].tolist()
        sim_state['demand_actual'] = int(
            self.simulator.skus.info.dynamic[0].iteration_actual_demand)
        # ["demand_forecast"][sim.simulator.period+1:sim.simulator.period+11].tolist()
        sim_state['demand_forecast'] = [
            int(i) for i in self.simulator.skus.info.dynamic[0].iteration_demand_forecast]
        # [sim.simulator.period+1:sim.simulator.period+11].tolist()
        sim_state['demand_sigma'] = [
            int(i) for i in self.simulator.skus.info.dynamic[0].iteration_forecast_sigma]
        sim_state["inventory"] = [
            int(i) for i in self.simulator.skus.info.dynamic[0].inventory]
        sim_state['leads'] = [
            int(i) for i in self.simulator.skus.info.dynamic[0].leads]
        sim_state["constraint_relaxation"] = int(self.constraint_relaxation)
        sim_state["mip_solver_status"] = self.mip_cost_config
        sim_state["sim_reward"] = float(self.sum_cost_action_freq)
        sim_state["missed_sale_to_inventory_cost_ratio"] = float(
            self.missed_sale_to_inventory_cost_ratio)
        print("sim_state:\n", sim_state)
        return sim_state

    def episode_start(self, config: Dict[str, Any]):
        """ Called at the start of each episode """

        if "mode" in config.keys():
            self.mode = "assess"
            print('choosing random sigma for assessment only')
            self.episode_sigma = random.randint(0, config["sigmax"])
            config["sigmax"] = self.episode_sigma
            self.constraint_coupling = bool(config["constraint_coupling"])
        else:
            self.mode = "train"
            config["number_of_sku"] = 1
            config["number_of_stages"] = 4
            self.constraint_coupling = bool(0)

        self.simulator = make_multi_sku_env(config=config)
        self.mip_solver = MipSolver(config=None)
        self.sum_cost_action_freq = 0
        self.action_frequency = config["action_frequency"]
        self.brain_action_previous = self.simulator.skus.n_levels*[0]
        self.missed_sale_to_inventory_cost_ratio = config["missed_sale_to_inventory_cost_ratio"]
        return None

    def episode_step(self, action: Dict[str, Any]):
        """ Called for each step of the episode """
        print('Brain action is:', action)
        if self.mode == "train":
            brain_action = [[action["safety_stock_stage0"],
                            action["safety_stock_stage1"],
                            action["safety_stock_stage2"]]]
        else:
            brain_action = action["skus_safety"]

        print(brain_action)
        if self.simulator.period % self.action_frequency == 0:
            print('using brain action:', self.iter_count)
            pass
        else:
            print('using previous action', self.simulator.period)
            brain_action = self.brain_action_previous
            print(f'brain action {brain_action}')
        self.sum_cost_action_freq = 0
        for i in range(0, self.action_frequency):
            self.previous_safety = brain_action
            if self.mode == "training":
                self.simulator.skus.info.dynamic[0].safety = brain_action[0]
            else:
                for sku in range(self.simulator.skus.sku_count):
                    self.simulator.skus.info.dynamic[sku].safety = brain_action[sku]
            skus = self.simulator.skus
            self.constraint_relaxation = 0
            self.mip_cost_config = "Hybrid"
            sim_action, _, _ = self.mip_solver.get_mip_action_multiple_sku(
                skus=self.simulator.skus, cost_function="Hybrid", constraint_coupling=self.constraint_coupling)
            if None in [item for sublist in sim_action for item in sublist]:
                self.mip_cost_config = "Hybrid"
                sim_action, _, _ = self.mip_solver.get_mip_action_multiple_sku(
                    skus=skus, cost_function="Hybrid", constraint_coupling=bool(0))
                self.constraint_relaxation = 1
                if None in [item for sublist in sim_action for item in sublist]:
                    self.mip_cost_config = "Infeasible"
                    self.constraint_relaxation = -1
                    sim_action = self.simulator.skus.sku_count * \
                        [self.simulator.skus.n_levels*[20]]

            self.simulator.step(sim_action)
            self.mip_action = sim_action
            self.sum_cost_action_freq += self.simulator.iteration_cost

    def halted(self) -> bool:
        """
        Should return True if the simulator cannot continue for some reason
        """

        # If arm hits rails of physical limit +/- 90 degrees
        return False

    def log_iterations_old(self, state, action, episode: int = 1, iteration: int = 1):
        """Log iterations during training to a CSV.
        Parameters
        ----------
        state : Dict
        action : Dict
        episode : int, optional
        iteration : int, optional
        """

        import pandas as pd

        def add_prefixes(d, prefix: str):
            return {f"{prefix}_{k}": v for k, v in d.items()}

        state = add_prefixes(state, "state")
        action = add_prefixes(action, "action")
        config = add_prefixes(self.config, "config")
        print('-----------------------------------')

        data = {**state, **action, **config}
        data["episode"] = episode
        data["iteration"] = iteration
        #print('data: \n', data)
        log_df = pd.DataFrame(data, index=[0])

        if episode == 1 and iteration == 1:
            # Store initial states and configs because we don't have actions yet
            print(
                'Collecting episode start logdf, waiting for action keys from episode step')
            self.initial_log = data
        elif iteration >= 2:
            if os.path.exists(self.log_full_path):
                # Check if we've already written to a file with 2 rows, continue writing
                log_df = pd.DataFrame(
                    {k: log_df[k] for k in self.desired_dict_order})
                log_df.to_csv(
                    path_or_buf=self.log_full_path, mode="a", header=False, index=False
                )
            else:
                # Take intial states and configs from ep 1, update with actions with None
                for key, val in action.items():
                    self.initial_log[key] = None
                self.initial_log = pd.DataFrame(self.initial_log, index=[0])
                self.action_keys = action.keys()

                log_df = pd.concat([self.initial_log, log_df], sort=False)
                log_df.to_csv(
                    path_or_buf=self.log_full_path, mode="w", header=True, index=False
                )
                if episode == 1 and iteration == 2:
                    self.desired_dict_order = log_df.keys()
        elif iteration == 1:
            # Now every episode start will use action keys and reorder dict properly
            for key in self.action_keys:
                log_df[key] = None
            log_df = pd.DataFrame({k: log_df[k]
                                  for k in self.desired_dict_order})
            log_df.to_csv(
                path_or_buf=self.log_full_path, mode="a", header=False, index=False
            )
        else:
            print('Something else went wrong with logs')
            exit()

    def log_iterations(self, state, action, episode: int = 1, iteration: int = 1):
        """Log iterations during training to a CSV.
        Parameters
        ----------
        state : Dict
        action : Dict
        episode : int, optional
        iteration : int, optional
        """

        import pandas as pd

        log_state = state.copy()
        log_action = action.copy() if action is not None else 0
        log_config = self.config.copy()
        for key, value in log_state.items():
            if type(value) == list:
                log_state[key] = str(log_state[key])

        for key, value in log_action.items():
            if type(value) == list:
                log_action[key] = str(log_action[key])

        for key, value in log_config.items():
            if type(value) == list:
                log_config[key] = str(log_config[key])
        data = {**log_state, **log_action, **log_config}
        data["episode"] = episode
        data["iteration"] = iteration
        data["TimeStamp"] = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")

        try:
            data["SessionIndex"] = self.SessionIndex
        except:
            # test loop with random or exported brain
            pass

        # print('data?????????????????????????????????????????????: \n', data)

        log_df = pd.DataFrame(data, index=[0])

        if episode == 1 and iteration == 1:
            # Store initial states and configs because we don't have actions yet
            print(
                "Collecting episode start logdf, waiting for action keys from episode step"
            )

            self.initial_log = data
        elif iteration >= 2:
            if os.path.exists(self.log_full_path):
                # Check if we've alrdy written to a file with 2 rows, continue writing
                log_df = pd.DataFrame(
                    {k: log_df[k] for k in self.desired_dict_order})
                log_df.to_csv(
                    path_or_buf=self.log_full_path, mode="a", header=False, index=False
                )
            else:
                # Take initial states and configs from ep 1, update with actions with None
                for key, val in action.items():
                    self.initial_log[key] = None
                self.initial_log = pd.DataFrame(self.initial_log, index=[0])
                self.action_keys = action.keys()

                log_df = pd.concat([self.initial_log, log_df], sort=False)
                # self.log_full_path = pathlib.Path(*self.log_full_path.parts[:-1]) / pathlib.Path(
                #     self.SessionIndex.replace('.', '_') + '__'+self.log_full_path.parts[-1])
                log_df.to_csv(
                    path_or_buf=self.log_full_path, mode="w", header=True, index=False
                )
                if episode == 1 and iteration == 2:
                    self.desired_dict_order = log_df.keys()
        elif iteration == 1:
            # Now every episode start will use action keys and reorder dict properly
            for key in self.action_keys:
                log_df[key] = None
            log_df = pd.DataFrame({k: log_df[k]
                                  for k in self.desired_dict_order})
            log_df.to_csv(
                path_or_buf=self.log_full_path, mode="a", header=False, index=False
            )
        else:
            print("Something else went wrong with logs")
            exit()


def env_setup(env_file: str = ".env"):
    """Helper function to setup connection with Project Bonsai
    Returns
    -------
    Tuple
        workspace, and access_key
    """

    load_dotenv(dotenv_path=env_file, verbose=True, override=True)
    workspace = os.getenv("SIM_WORKSPACE")
    access_key = os.getenv("SIM_ACCESS_KEY")

    env_file_exists = os.path.exists(env_file)
    if not env_file_exists:
        open(env_file, "a").close()

    if not all([env_file_exists, workspace]):
        workspace = input("Please enter your workspace id: ")
        set_key(env_file, "SIM_WORKSPACE", workspace)
    if not all([env_file_exists, access_key]):
        access_key = input("Please enter your access key: ")
        set_key(env_file, "SIM_ACCESS_KEY", access_key)

    load_dotenv(dotenv_path=env_file, verbose=True, override=True)
    workspace = os.getenv("SIM_WORKSPACE")
    access_key = os.getenv("SIM_ACCESS_KEY")

    return workspace, access_key


def test_policy(
    render: bool = False,
    num_iterations: int = 20,
    log_iterations: bool = False,
    policy=zero_policy,
    policy_name: str = "zero_policy",
    scenario_file: str = "assess_config.json",
):
    """Test a policy using random actions over a fixed number of episodes
    Parameters
    ----------
    render : bool, optional
        Flag to turn visualization on
    """

    # Use custom assessment scenario configs
    with open(scenario_file) as fname:
        assess_info = json.load(fname)
    print("assess_info")
    print(assess_info)
    scenario_configs = assess_info['episodeConfigurations']
    num_episodes = assess_info['number_of_episodes'] + 1
    num_iterations = assess_info['number_of_episodes']

    current_time = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    log_file_name = current_time + "_" + policy_name + "_log.csv"
    sim = TemplateSimulatorSession(
        render=render, log_data=log_iterations, log_file_name=log_file_name
    )
    for episode in range(1, num_episodes):
        iteration = 1
        terminal = False
        sim_state = sim.episode_start(config=scenario_configs)
        sim_state = sim.get_state()

        print('policy name: ', policy_name)
        if (policy_name != 'random' and policy_name != 'safety_policy' and policy_name != 'zero_policy'
                and policy_name != 'random_safety_policy'):
            print('using exported brain')

            if any('exported_brain_url' in key for key in policy.keywords):
                # Reset the Memory vector because exported brains don't understand episodes
                url = '{}/v1'.format(policy.keywords['exported_brain_url'])
                forget_memory(url)

        if log_iterations:
            sim.config = scenario_configs
            #sim.log_iterations(sim_state, {}, episode, iteration)
        print(f"Running iteration #{iteration} for episode #{episode}")
        #iteration += 1
        print('num_iterations:', num_iterations)
        while not terminal:
            action = {"safety_stock_stage0": [],
                      "safety_stock_stage1": [],
                      "safety_stock_stage2": [],
                      "skus_safety": []}
            states_log = {}
            states_log["demand_actual"] = []
            states_log["demand_sigma"] = []
            states_log["demand_forecast"] = []
            states_log["transit_orders"] = []
            states_log["leads"] = []
            states_log["skus_missed_sale_to_inventory_cost_ratio"] = []

            for sku in range(sim.simulator.skus.sku_count):
                print('sku number is ..........................>>>>>>>>>>>>>>>', sku)
                transit_orders = []
                for level in range(sim.simulator.skus.n_levels):
                    transit_orders.append(
                        list(sim.simulator.skus.info.dynamic[sku].transit_order[level])[0:15])
                transit_orders = [
                    item for sublist in transit_orders for item in sublist]
                # make sure this matches that of platform
                sim.simulator.skus.info.dynamic[sku].forecast_window = 15
                sim_state = {}
                sim_state['transit_orders'] = [
                    int(order) for order in transit_orders]
                # ["demand_actual"][sim.simulator.period].tolist()
                sim_state['demand_actual'] = int(
                    sim.simulator.skus.info.dynamic[sku].iteration_actual_demand)
                # ["demand_forecast"][sim.simulator.period+1:sim.simulator.period+11].tolist()
                sim_state['demand_forecast'] = [
                    int(i) for i in sim.simulator.skus.info.dynamic[sku].iteration_demand_forecast]
                # [sim.simulator.period+1:sim.simulator.period+11].tolist()
                sim_state['demand_sigma'] = [
                    int(i) for i in sim.simulator.skus.info.dynamic[sku].iteration_forecast_sigma]
                sim_state["inventory"] = [
                    int(i) for i in sim.simulator.skus.info.dynamic[sku].inventory]
                sim_state['leads'] = [
                    int(i) for i in sim.simulator.skus.info.dynamic[sku].leads]
                sim_state["constraint_relaxation"] = int(
                    sim.constraint_relaxation)
                sim_state["mip_solver_status"] = sim.mip_cost_config
                sim_state["sim_reward"] = float(sim.sum_cost_action_freq)
                sim_state["missed_sale_to_inventory_cost_ratio"] = float(
                    sim.simulator.skus.info.generic[sku].missed_sale_cost[0]/sim.simulator.skus.info.generic[sku].storage_cost[0])
                # ["demand_actual"][sim.simulator.period].tolist()
                sim_state['demand_actual'] = sim.simulator.skus.info.dynamic[sku].iteration_actual_demand
                # ["demand_forecast"][sim.simulator.period+1:sim.simulator.period+11].tolist()
                sim_state['demand_forecast'] = sim.simulator.skus.info.dynamic[sku].iteration_demand_forecast
                # [sim.simulator.period+1:sim.simulator.period+11].tolist()
                sim_state['demand_sigma'] = sim.simulator.skus.info.dynamic[sku].iteration_forecast_sigma

                states_log["transit_orders"].append(transit_orders)
                states_log["demand_actual"].append(
                    sim.simulator.skus.info.dynamic[sku].iteration_actual_demand)
                states_log["demand_sigma"].append(
                    sim.simulator.skus.info.dynamic[sku].iteration_forecast_sigma)
                states_log["demand_forecast"].append(
                    sim.simulator.skus.info.dynamic[sku].iteration_demand_forecast)
                # states_log["transit_orders"].append(sim.simulator.skus.info.dynamic[sku].transit_order)
                states_log["leads"].append(
                    sim.simulator.skus.info.dynamic[sku].leads)
                states_log["skus_missed_sale_to_inventory_cost_ratio"].append(int(
                    sim.simulator.skus.info.generic[sku].missed_sale_cost[0]/sim.simulator.skus.info.generic[sku].storage_cost[0]))
                # fix json serialization issue with numpy int32
                for item, value in sim_state.items():
                    # print(sim_state)
                    if type(value) == list:
                        sim_state[item] = [int(i) for i in value]
                print('sim state :\n', sim_state)
                sku_action = policy(sim_state)
                print(f'sku action for sku {sku} .......>>>> :\n', sku_action)
                brain_action = [sku_action["safety_stock_stage0"],
                                sku_action["safety_stock_stage1"],
                                sku_action["safety_stock_stage2"]]
                action["safety_stock_stage0"].append(brain_action[0])
                action["safety_stock_stage1"].append(brain_action[1])
                action["safety_stock_stage2"].append(brain_action[2])
                action['skus_safety'].append(brain_action)
                # update safety
                #sim.simulator.skus.info.dynamic[sku].safety = brain_action

            # passing empty actions as sku handles safety

            # po[sku] = purchase orders for all levels
            print('action is>>>>>>>>>>>>>>>>>>>>>:\n ', action)
            sim.episode_step(action)

            # add mip action to states
            states_log["constraint_relaxation"] = int(
                sim.constraint_relaxation)
            states_log["mip_solver_status"] = sim.mip_cost_config
            states_log["sim_reward"] = float(sim.sum_cost_action_freq)
            states_log["mip_action"] = sim.mip_action
            states_log['iter_cost'] = sim.simulator.iteration_cost.tolist()
            states_log["inventory"] = sim.simulator.current_inventory_level.tolist()
            states_log["missed_sale"] = sim.simulator.iteration_lost_sale.tolist()
            states_log["replenishment"] = sim.simulator.iteration_replenishment.tolist()
            states_log["state_in_transit"] = sim.simulator.state_in_transit.tolist()
            states_log["demand_all_levels"] = sim.simulator.Demand_all_levels.tolist()
            states_log["hold_cost_per_unit"] = sim.simulator.holding_costs.tolist()
            states_log["inventory_constraint"] = sim.simulator.skus.constraints.inventory_capacity
            states_log["episode_sigmax"] = sim.episode_sigma
            states_log["inventory_sum"] = np.sum(
                sim.simulator.current_inventory_level, axis=0).tolist()
            states_log["inventory_sum_plus_demand"] = (np.sum(sim.simulator.current_inventory_level, axis=0) +
                                                       np.sum(sim.simulator.Demand_all_levels)).tolist()
            states_log["constraint_relaxed"] = sim.constraint_relaxation
            states_log["mip_cost_config"] = sim.mip_cost_config
            print(states_log)
            sim_state = states_log
            if log_iterations:
                sim.log_iterations(sim_state, action, episode, iteration)
            print(f"Running iteration #{iteration} for episode #{episode}")
            print(f"Observations: {sim_state}")
            print(f"Actions: {action}")
            iteration += 1
            terminal = iteration >= num_iterations+2 or sim.halted()

    return sim


def main(
    render: bool = False,
    log_iterations: bool = False,
    config_setup: bool = False,
    env_file: Union[str, bool] = ".env",
    workspace: str = None,
    accesskey: str = None,
):
    """Main entrypoint for running simulator connections
    Parameters
    ----------
    render : bool, optional
        visualize steps in environment, by default True, by default False
    log_iterations: bool, optional
        log iterations during training to a CSV file
    config_setup: bool, optional
        if enabled then uses a local `.env` file to find sim workspace id and access_key
    env_file: str, optional
        if config_setup True, then where the environment variable for lookup exists
    workspace: str, optional
        optional flag from CLI for workspace to override
    accesskey: str, optional
        optional flag from CLI for accesskey to override
    """

    # check if workspace or access-key passed in CLI
    use_cli_args = all([workspace, accesskey])

    # use dotenv file if provided
    use_dotenv = env_file or config_setup

    # check for accesskey and workspace id in system variables
    # Three scenarios
    # 1. workspace and accesskey provided by CLI args
    # 2. dotenv provided
    # 3. system variables
    # do 1 if provided, use 2 if provided; ow use 3; if no sys vars or dotenv, fail

    if use_cli_args:
        # BonsaiClientConfig will retrieve as environment variables
        os.environ["SIM_WORKSPACE"] = workspace
        os.environ["SIM_ACCESS_KEY"] = accesskey
    elif use_dotenv:
        if not env_file:
            env_file = ".env"
        print(
            f"No system variables for workspace-id or access-key found, checking in env-file at {env_file}"
        )
        workspace, accesskey = env_setup(env_file)
        load_dotenv(env_file, verbose=True, override=True)
    else:
        try:
            workspace = os.environ["SIM_WORKSPACE"]
            accesskey = os.environ["SIM_ACCESS_KEY"]
        except:
            raise IndexError(
                f"Workspace or access key not set or found. Use --config-setup for help setting up."
            )

    # Grab standardized way to interact with sim API
    sim = TemplateSimulatorSession(render=render, log_data=log_iterations)

    # Configure client to interact with Bonsai service
    config_client = BonsaiClientConfig()
    client = BonsaiClient(config_client)

    # Load json file as simulator integration config type file
    try:
        with open("interface.json") as file:
            interface = json.load(file)
    except:
        interface = {
            "name": "IMS",
            "timeout": 60,
            "description": {
                "config": {"empty": 0},
                "action": {"empty": 0},
                "state": {"empty": 0}
            }
        }

    # Create simulator session and init sequence id
    registration_info = SimulatorInterface(
        name=sim.env_name,
        timeout=interface["timeout"],
        simulator_context=config_client.simulator_context,
        description=interface["description"],
    )

    def CreateSession(
        registration_info: SimulatorInterface, config_client: BonsaiClientConfig
    ):
        """Creates a new Simulator Session and returns new session, sequenceId
        """

        try:
            print(
                "config: {}, {}".format(
                    config_client.server, config_client.workspace)
            )
            registered_session: SimulatorSessionResponse = client.session.create(
                workspace_name=config_client.workspace, body=registration_info
            )
            print("Registered simulator. {}".format(
                registered_session.session_id))

            return registered_session, 1
        except HttpResponseError as ex:
            print(
                "HttpResponseError in Registering session: StatusCode: {}, Error: {}, Exception: {}".format(
                    ex.status_code, ex.error.message, ex
                )
            )
            raise ex
        except Exception as ex:
            print(
                "UnExpected error: {}, Most likely, it's some network connectivity issue, make sure you are able to reach bonsai platform from your network.".format(
                    ex
                )
            )
            raise ex

    registered_session, sequence_id = CreateSession(
        registration_info, config_client)
    episode = 0
    iteration = 1

    try:
        while True:
            # Advance by the new state depending on the event type
            # TODO: it's risky not doing doing `get_state` without first initializing the sim
            sim_state = SimulatorState(
                sequence_id=sequence_id, state=sim.get_state(), halted=sim.halted(),
            )
            try:
                event = client.session.advance(
                    workspace_name=config_client.workspace,
                    session_id=registered_session.session_id,
                    body=sim_state,
                )
                sequence_id = event.sequence_id
                print(
                    "[{}] Last Event: {}".format(
                        time.strftime("%H:%M:%S"), event.type)
                )
            except HttpResponseError as ex:
                print(
                    "HttpResponseError in Advance: StatusCode: {}, Error: {}, Exception: {}".format(
                        ex.status_code, ex.error.message, ex
                    )
                )
                # This can happen in network connectivity issue, though SDK has retry logic, but even after that request may fail,
                # if your network has some issue, or sim session at platform is going away..
                # So let's re-register sim-session and get a new session and continue iterating. :-)
                registered_session, sequence_id = CreateSession(
                    registration_info, config_client
                )
                continue
            except Exception as err:
                print("Unexpected error in Advance: {}".format(err))
                # Ideally this shouldn't happen, but for very long-running sims It can happen with various reasons, let's re-register sim & Move on.
                # If possible try to notify Bonsai team to see, if this is platform issue and can be fixed.
                registered_session, sequence_id = CreateSession(
                    registration_info, config_client
                )
                continue

            # Event loop
            if event.type == "Idle":
                time.sleep(event.idle.callback_time)
                print("Idling...")
            elif event.type == "EpisodeStart":
                print(event.episode_start.config)
                sim.episode_start(event.episode_start.config)
                episode += 1
                if sim.log_data:
                    sim.log_iterations(
                        episode=episode,
                        iteration=iteration,
                        state=sim.get_state(),
                        action={},
                    )
            elif event.type == "EpisodeStep":
                sim.episode_step(event.episode_step.action)
                iteration += 1
                if sim.log_data:
                    sim.log_iterations(
                        episode=episode,
                        iteration=iteration,
                        state=sim.get_state(),
                        action=event.episode_step.action,
                    )
            elif event.type == "EpisodeFinish":
                print("Episode Finishing...")
                iteration = 1
            elif event.type == "Unregister":
                print(
                    "Simulator Session unregistered by platform because '{}', Registering again!".format(
                        event.unregister.details
                    )
                )
                registered_session, sequence_id = CreateSession(
                    registration_info, config_client
                )
                continue
            else:
                pass
    except KeyboardInterrupt:
        # Gracefully unregister with keyboard interrupt
        client.session.delete(
            workspace_name=config_client.workspace,
            session_id=registered_session.session_id,
        )
        print("Unregistered simulator.")
    # except Exception as err:
    #     # Gracefully unregister for any other exceptions
    #     client.session.delete(
    #         workspace_name=config_client.workspace,
    #         session_id=registered_session.session_id,
    #     )
    #     print("Unregistered simulator because: {}".format(err))


if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser(
        description="Bonsai and Simulator Integration...")
    parser.add_argument(
        "--render", action="store_true", default=False, help="Render training episodes",
    )
    parser.add_argument(
        "--log-iterations",
        action="store_true",
        default=False,
        help="Log iterations during training",
    )
    parser.add_argument(
        "--config-setup",
        action="store_true",
        default=False,
        help="Use a local environment file to setup access keys and workspace ids",
    )
    parser.add_argument(
        "--env-file",
        type=str,
        metavar="ENVIRONMENT FILE",
        help="path to your environment file",
        default=None,
    )
    parser.add_argument(
        "--workspace",
        type=str,
        metavar="WORKSPACE ID",
        help="your workspace id",
        default=None,
    )
    parser.add_argument(
        "--accesskey",
        type=str,
        metavar="Your Bonsai workspace access-key",
        help="your bonsai workspace access key",
        default=None,
    )

    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--test-local",
        action="store_true",
        help="Run simulator locally with a random policy, without connecting to platform",
    )

    group.add_argument(
        "--test-exported",
        type=int,
        const=5000,  # if arg is passed with no PORT, use this
        nargs="?",
        metavar="PORT",
        help="Run simulator with an exported brain running on localhost:PORT (default 5000)",
    )

    parser.add_argument(
        "--iteration-limit",
        type=int,
        metavar="EPISODE_ITERATIONS",
        help="Episode iteration limit when running local test.",
        default=200,
    )

    parser.add_argument(
        "--custom-assess",
        type=str,
        default=None,
        help="Custom assess config json filename",
    )

    # parser.add_argument(
    #     "--reuse-brain",
    #     type=str,
    #     default=False,
    #     help="use brain in a loop for multiple actions",
    # )

    args, _ = parser.parse_known_args()

    scenario_file = 'assess_config.json'
    if args.custom_assess:
        scenario_file = args.custom_assess

    if args.test_local:
        test_policy(
            render=args.render, log_iterations=args.log_iterations, policy=zero_policy
        )
    elif args.test_exported:
        port = args.test_exported
        url = f"http://localhost:{port}"
        print(f"Connecting to exported brain running at {url}...")
        trained_brain_policy = partial(brain_policy, exported_brain_url=url)
        test_policy(
            render=args.render,
            log_iterations=args.log_iterations,
            policy=trained_brain_policy,
            policy_name="exported",
            num_iterations=args.iteration_limit,
            scenario_file=scenario_file,
        )
    else:
        # test_policy(
        #     render=args.render, log_iterations=args.log_iterations, policy=random_policy
        # )
        main(
            config_setup=args.config_setup,
            render=args.render,
            log_iterations=args.log_iterations,
            env_file=args.env_file,
            workspace=args.workspace,
            accesskey=args.accesskey,
        )
