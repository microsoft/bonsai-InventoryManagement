# Inventory Management using hybrid approach based on brain (deep reinforcement learning) and classical optimization method (mixed integer programming) 

# Prerequsite:
Create supplychain conda envionment and activate it
```
conda env update -f environment.yml
conda activate supplychain
```

## Bussiness problem

A retailer faces uncertain consumer demand from day to day and need to hold inventory levels to meet costumer demand. He deals with 100s of products. High inventory levels have holding costs and low inventory levels would not satisfy customer needs and results in missed sale cost. Additionally, customer faces some constraints, such as total inventory capacity. Additionally, each product has a different missed sale to inventory cost ratio. The Goal is to keep inventory levels at an optimal levels, such that it results in a good trade-off between service level and holding costs and meets total inventory capacity as much as possible.

Periodically, the retailer wishes to make optimal decisions on safety stock levels and periodic purchase order levels for all the stages of the supply chain. The retailor has access to demand forecast at customer level. However, the future forecasts are uncertain and depend on variable forecast confidence intervals, i.e forecasts are presented through mean and standard deviation. 

The retailer's supply chain resembles a multi-echelon inventory system, where all the products are resources from same distribution/manufacturing centers. The following image shows a multi-echelon inventory system. Each stage has inventory holding areas and capacitated production area, except the stage0. Retails

<img src = "img/MultiEchelonSupplyChain.png" alt= "drawing" width="500"/>

[reference: Hubbs et al.: OR-Gym: A Reinforcement Learning Library]

We continue with M = 3. As and example, the retailor is located in New York. He makes product purchases from a distribution center in Tennessee, with a lead time of 2 days. The Tennessee distribution center makes the purchases from another distribution center in Los Angeles with a lead time of 3 days. Subsequently, this LA distributor makes purchases from a manufacturer in China with a lead time of 4 days. The manufacturer has access to unlimited supply of raw materials. 

### Solution architecture

We propose a hybrid approach due to the following reasons:

(1) 100s of products with constraints leads to curse of dimensionality where AI only solutions becomes cumbersome if not practically impossible to train. 

(2) On the other hand, classical optimization methods becomes practically difficult/impossible to tackle uncertainty in a time efficient way. Searching for an optimal solution may take very long time due to demand forecast uncertainties. Not to mention that other aspects of supply chain such as lead time may become uncertain as well, limiting deterministic classical search methods inefficient to solve the problem. 

As an alternative, we will use brain as a de-fuzzifier that makes a crisp decision on ideal safety stock levels for each product without considering any cross-product constraints. Then classical optimization method of mixed integer programming can be applied. In this approach, We train brain for products with different missed sale to inventory holding cost ratio (e.g. 10, 100, 1000). Once the brains are trained, we use the policies to specify ideal future safety stock level for each product. Then, we construct desired future stock levels as follows: forecast mean + anticipated ideal safety stock levels. Finally, we use mip solver to make final adjustments on the purchase order levels, while satisfying constraints.  

The below image summarizes the approach:

<img src = "img/SolutionArchitecture_train_deploy.png" alt= "drawing" width="1000"/>

### Objective
Minimize cost of missing demand + cost of holding for each single product. 


|                    | Defintion  | Notes |
| -------------------| -----------|-------|
|Observations        |transit orders, demand_forecast, demand_sigma, current_demand, inventory, leads, misses_sale_to_inventory_cost_ratio|demand_sigma: standard deviation of the forecasts|
|Actions | safety stock levels for each stage|
|Control frequency| user defined | retailor can decide on time interval of adjusting the future expected safety stock levels  
|Episode Configuration| demand profile parameters, sigmax, offset, amp1, ph1, amp2, ph2, randvar, lead profile, missed_sale_to_inventory_cost, action frequency | custom demand is parameterized to manage demand properties such as (1) sigmax: max uncertainty threshold (2)offset: offsets demand levels, amp1,ph1, amp2, ph2: non-stationarity, randvar: baseline demand noise. Various lead profiles are available for more complex situations such as variable lead times, etc|

An example demand profile consumed by the simulator is shown below showing actual and forecasted demand. Note seasonality, noise, and forecast uncertainty. 

<img src = "img/CustomDemand.png" alt= "drawing" width="600"/>

### Sim packages (RL plus mip)

To build bonsai sim package, login using azure-cli and then run the following command.

```
az acr build --image IMAGENAME --registry YOURBONSAIREGISTRY .
```

### Brain training

Use the following inkling file in bonsai platform to train brain that handles a single sku with wide range of missed sale to inventory cost ratio. You may also chose a separate brain for a specific missed sale to inventory cost ratio.

To train brain for specif cost ratio, use the following inkling file and modify "cost_ratio" constant near top of the file. The default is cost_ratio = 100. For this tutorial, use the following inkling file.

```
MachineTeacher_const_cost_ratio.ink
```

In order to train for different cost ratio using single brain training, you may use the following inkling. 

```
MachineTeacher_range_cost_ratio.ink
```

To train for a generic use case with variable cost_ratio and lead time, please refer to the following inkling. Please note that we have not done any experiments to support results with variable lead time, yet. Feel free to experiment.

```
MachineTeacher_generic.ink
```
Note that brain training is only required for single sku. For multi-sku, We will export trained single-sku brain and use it within a loop over all the skus. Then the mip solver optimizes the final purchase orders while respecting overall capacity constraints. 

### Assessment logs

To create assessment logs for 

(1) multi-sku mip plus brain approach:
first export trained brain and then run
```
python main_assess_mip_plus_brain_or_other_safety_policy.py --test-exported --test-exported <PORT> --log-iterations
```
(2) multi-sku mip only approach:
run the following command 
```
python main_assess_mip_only.py --test-local --log-iterations 
```
Take note of "assess_config.json" for details of the experiment, such as number of skus and total inventory capacity.
Also make sure to run the assessments long enough (over 1000 episodes) to account for variabilities due to stochasticity of the environment and for statistically significant results.

## Results and Analysis
### Single SKU brain training (step1 in solution architecture) and assessment results:
We studied the effects of demand forecast uncertainty and ratio of missed sale cost to inventory holding (cost ratio). Conservatively, we trained different brains for different cost ratio. A sample brain training graph is shown below.
<img src = "img/Brain_trained_cost_ratio10.png" alt= "drawing" width="800"/>

The image below shows total cost vs level of uncertainty for different cost ratios. Brain plus mip outperforms classical optimizer when cost ratio is higher. This makes sense as higher missed sales cost accentuates the need for safety stock levels. When cost ratio is equal to one, there is no need to hold safety stock level as holding extra inventory is as costly as missing sale.

<img src = "img/SingleSKU_cost_ratio.png" alt= "drawing" width="800"/>

Refer to the jupyter notebookes for details.
Note: log files are not included due to their large sizes.  
### Assessment on multisku (step2 in solution architecture):
Following the procedure shown in step2 of the solution architecture, brain plus mip outperforms mip only approach. A sample comparison is shown below.

<img src = "img/MultiSKU.png" alt= "drawing" width="800"/>

Refer to the jupyter notebooks in log folder for details. 
Note: log files are not included due to their large sizes. 

### Developer guide
For a similar multi-sku problem, make sure to modify essential files, such as chain.definition.py, constraints.py. Then modify the SKUInfoFactory in multi_sku.py. You may also create a new factory following similar template, but make sure to import and replace SKUInfoFactory with your costume SKUInfoFactory, in relevant files such as inventory_management.py  
For different demand and lead profiles, make sure to develop the code in utils_demand.py and utils.py
