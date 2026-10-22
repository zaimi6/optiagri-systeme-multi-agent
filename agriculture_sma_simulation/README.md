# Multi-Agent System for Moroccan Agricultural Optimization Using Real Official Data

## Objective

This university project presents a Python multi-agent simulator for Moroccan agricultural optimization. It uses real official Moroccan agricultural datasets as the base, derives a small set of transparent simulation variables, applies fuzzy logic for softer agronomic reasoning, compares multiple crop-allocation scenarios, and opens a live Pygame desktop interface for presentation.

## Academic Framing

The project is designed for a Mathematics for AI context. It is an explainable simulation and decision-support model, not a black-box predictor and not a production agronomic planning system. The emphasis is on clear agent roles, interpretable formulas, controlled scaling, and live demonstration quality.

## Real Moroccan Datasets Used

The project is built around local official Moroccan Excel files placed in `data/`:

- `Production vegetale 2010-2022`
- `Indicateurs sectoriels - Agriculture et peche 2010-2020`

### Dataset Roles

1. `Production vegetale 2010-2022`

- crop production observations
- cultivated surface values
- crop and filiere names
- multi-year crop yield baselines after cleaning

2. `Indicateurs sectoriels - Agriculture et peche 2010-2020`

- precipitation indicators
- cultivated area indicators
- sector context used to derive normalized resource scores

## Real Variables and Derived Variables

The project clearly separates what comes directly from the official files and what is derived for simulation.

### Real Official Variables

- crop names
- filiere names
- year or occurrence labels
- production quantities
- cultivated surface
- precipitation indicators
- cultivated area indicators

### Derived Variables

Derived variables are explicitly labeled in the outputs and logs:

- `soil_quality_index_derived`
- `water_availability_index_derived`
- `climate_favorability_index_derived`
- `fertilizer_efficiency_index_derived`
- `resource_balance_index_derived`
- `decision_surface_ha_derived`
- `yield_reference_t_ha_derived`
- normalized crop profile scores
- fuzzy suitability scores
- optimization scores

The simulator does not claim that these variables come directly from the official Moroccan files.

## Agent Architecture

### `CoordinatorAgent`

- orchestration only
- loads data through `DataAgent`
- runs the simulation workflow in the correct order
- updates the live interface
- saves outputs

### `DataAgent`

- loads the local Moroccan Excel files
- cleans and restructures the available columns dynamically
- merges usable agricultural information
- preserves the distinction between real and derived variables

### `SoilAgent`

- computes normalized soil suitability
- supports crop compatibility evaluation

### `WaterAgent`

- computes normalized water sufficiency
- penalizes water-intensive crops when water is limited

### `ClimateAgent`

- computes normalized climate favorability
- supports crop-year compatibility

### `CropAgent`

- manages crop-specific logic
- evaluates compatibility with the current multi-agent context

### `FertilizerAgent`

- estimates fertilizer contribution
- keeps fertilizer logic explicitly marked as derived when official fertilizer columns are absent

### `FarmTypeAgent`

- selects an appropriate farm strategy such as `rainfed`, `irrigated`, `intensive`, or `mixed`
- influences efficiency and resource fit

### `FuzzyLogicAgent`

- applies simple explainable fuzzy rules over soil, water, climate, crop compatibility, fertilizer, and farm strategy outputs
- converts rigid numeric thresholds into softer agricultural reasoning

### `YieldAgent`

- combines normalized agent outputs and fuzzy adjustment
- computes expected yield and expected production with bounded scaling

### `OptimizationAgent`

- generates realistic allocation scenarios
- compares single-crop and mixed allocations
- ranks scenarios using expected output, yield, fuzzy suitability, and resource balance

### `DecisionAgent`

- transforms optimization results into readable academic summaries
- explains why the winning scenario was selected

### `VisualizationAgent`

- opens a real live Pygame desktop window automatically
- shows the agent flow, fuzzy logic stage, optimizer stage, current scenario, and ranking
- keeps the demo in fixed windowed mode for presentation

## Simulation Logic

The simulation follows this order:

1. `DataAgent` loads and prepares the official Moroccan datasets
2. `SoilAgent` computes soil suitability
3. `WaterAgent` computes water sufficiency
4. `ClimateAgent` computes climate favorability
5. `CropAgent` computes crop compatibility
6. `FertilizerAgent` computes fertilizer contribution
7. `FarmTypeAgent` selects the farm strategy
8. `FuzzyLogicAgent` applies fuzzy adjustment
9. `YieldAgent` computes expected yield and expected production
10. `OptimizationAgent` ranks crop-allocation scenarios
11. `DecisionAgent` explains the final recommendation
12. `VisualizationAgent` displays the process live

## Mathematical Formulation

The base yield model remains explainable:

```text
expected_yield ~= base_yield * moderated_multiplier
```

where the moderated multiplier is built from:

- normalized soil score
- normalized water score
- normalized climate score
- normalized fertilizer score
- normalized farm strategy score
- normalized crop compatibility score
- fuzzy suitability modifier

For allocation:

```text
total_expected_output = sum(allocated_surface_i * expected_yield_i)
```

The optimizer then ranks scenarios with a bounded composite score using:

- normalized expected output
- normalized expected yield
- normalized fuzzy suitability
- normalized resource balance
- light diversification bonus
- water-stress penalty

## Fuzzy Logic Layer

The fuzzy layer uses simple, readable sets such as:

- soil quality: low / medium / high
- water level: low / medium / high
- climate condition: low / medium / high
- crop compatibility: low / medium / high
- fertilizer efficiency: low / medium / high

Example rule behavior:

- if water is low and crop demand is high, penalize strongly
- if soil is good and climate is favorable, boost suitability
- if fertilizer is medium but soil is poor, only partial recovery is allowed
- if crop compatibility and farm fit are both strong, increase confidence

## Scenario Optimization

The optimizer evaluates scenarios such as:

- 100 percent single-crop allocations
- 50/50 two-crop mixes
- 70/30 and 60/40 two-crop mixes
- selected three-crop mixes when enough crops are available

This allows the system to recommend whether a single crop or a diversified land split is better under the current Moroccan agricultural context.

## Interface

The live interface includes:

- a fixed windowed Pygame window
- a main simulation area with visible agent nodes
- an explicit fuzzy logic node
- an explicit optimizer node
- a right panel highlighting best crop, best allocation, expected yield, expected output, fuzzy score, and resource balance
- a lower comparison panel showing ranked scenarios

## How to Place Source Files

Place the official Moroccan Excel files in:

```text
agriculture_sma_simulation/data/
```

The project loads them locally and does not download anything from the internet.

## How to Run

From the project directory:

```bash
python main.py
```

The program loads the local Moroccan data, runs the multi-agent simulation, opens the live window automatically, and saves outputs in `results/`.

## Expected Outputs

The project generates:

- `results/merged_agriculture_dataset.csv`
- `results/scenario_results.csv`
- `results/best_allocation_report.txt`
- `results/simulation_summary.json`
- `results/agents_simulation.png`

## Limitations

- the official files are aggregate and not parcel-level datasets
- some simulation variables are derived because the official files do not contain every agronomic input needed for a live multi-agent model
- the fuzzy layer is intentionally simple for academic explainability
- the optimizer is scenario-based rather than a full continuous mathematical programming solver

## Conclusion

This project demonstrates a coherent multi-agent agricultural optimization simulator built on real Moroccan official datasets, explainable derived variables, fuzzy reasoning, allocation optimization, and a live desktop interface suitable for a university defense or professor demo.
