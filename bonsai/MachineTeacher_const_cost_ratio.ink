inkling "2.0"
using Number
using Math
using Goal

const action_frequency_days = 10 # in days 
const max_iteration: number = Math.Floor(200/action_frequency_days)
const cost_ratio = 100
type SimState {
    # sim observation: current order + 10 previous orders for 3 stages  
    transit_orders: number<0 .. 100>[45], 
    sim_reward: number<-1000 .. 1000>,
    demand_actual: number, 
    demand_forecast: number[15],
    demand_sigma: number[15],
    inventory: number[3],
    leads: number[3],
    missed_sale_to_inventory_cost_ratio: number,
}

type ObservableState {
    transit_orders: number<0 .. 3000>[45], 
    demand_actual: number, 
    demand_forecast: number[15],
    demand_sigma: number[15],
    inventory: number[3],
    
}

function GetObservableStates(State:SimState) : ObservableState{
    return{
        transit_orders: State.transit_orders,
        demand_forecast: State.demand_forecast,
        demand_sigma: State.demand_sigma,
        inventory: State.inventory,
        demand_actual: State.demand_actual,
        leads: State.leads, 
        missed_sale_to_inventory_cost_ratio:State.missed_sale_to_inventory_cost_ratio
    }
}

type SimAction {
    safety_stock_stage0: number<0 .. 10 step 1>,
    safety_stock_stage1: number<0 .. 10 step 1>,
    safety_stock_stage2: number<0 .. 10 step 1>
}

type SimConfig {

    sigmax: number<0 .. 12 step 1>,
    offset: number<0 .. 25 step 1>,
    amp1: number<0..10 step 1>,
    ph1: number<0..0.06 step 0.01>,
    amp2: number<0..5 step 1>,
    ph2: number<0..0.1 step 0.05>,
    randvar: number<1 ..20 step 1>,
    lead_profile: number<1 .. 9 step 1>, # 1 fixed, 2,3 variable 
    missed_sale_to_inventory_cost_ratio: number<10 .. 1000>,
    action_frequency: number, 
}

# Reward definiton
function GetReward(State: SimState) {
   # with long episodes inventory[0] tend to go to very high level during exploration 
    if State.inventory[0]>1000{
        return -100
    }
    else{
        return State.sim_reward/(100*action_frequency_days)
    }
}

function GetTerminal(State: SimState): Number.Bool {
    return State.inventory[0]>1000
}

simulator orgym_sim(action: SimAction, config: SimConfig): SimState {
    #package "imsreusable06172022-1"
}

graph (input: ObservableState) {
    concept OptimLevel(input): SimAction {
        curriculum {
            algorithm {
                Algorithm: "PPO",
                MemoryMode: "none",
            }
            source orgym_sim
            reward GetReward
            terminal GetTerminal
            state GetObservableStates
            training {
                # Limit episodes to 20 iterations (default value 1000)
                EpisodeIterationLimit: max_iteration,
                NoProgressIterationLimit: 3000000
            }
            lesson `lesson 1`{
                scenario {
                    lead_profile: 1,
                    sigmax:number<1 ..3 step 1>,
                    offset:20,
                    amp1:5,
                    ph1:0.02,
                    amp2:3,
                    ph2:0.05,
                    randvar:1,
                    action_frequency: action_frequency_days,
                    missed_sale_to_inventory_cost_ratio: cost_ratio,
                }

            }
            lesson `lesson 2`{
                scenario {
                    lead_profile: 1, 
                    sigmax:number<1 ..5 step 1>,
                    offset:20,
                    amp1:5,
                    ph1:0.02,
                    amp2:3,
                    ph2:0.05,
                    randvar:1,
                    action_frequency: action_frequency_days,
                    missed_sale_to_inventory_cost_ratio: cost_ratio,
                }
            }
            lesson `lesson 3`{
                scenario {
                    lead_profile: 1,
                    sigmax:number<1 ..10 step 1>,
                    offset:20,
                    amp1:5,
                    ph1:0.02,
                    amp2:3,
                    ph2:0.05,
                    randvar:1,
                    action_frequency: action_frequency_days,
                    missed_sale_to_inventory_cost_ratio: cost_ratio,
                }
            }
        }
    }
}