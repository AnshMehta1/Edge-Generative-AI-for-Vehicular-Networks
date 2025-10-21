import random
import torch

def weighted_aggregate_state_dicts(state_dicts, weights):
    """Weighted average aggregation of model parameters."""
    total = sum(weights)
    keys = state_dicts[0].keys()
    agg = {}
    for k in keys:
        agg[k] = sum(sd[k].float() * (w/total) for sd, w in zip(state_dicts, weights))
        agg[k] = agg[k].to(state_dicts[0][k].dtype)
    return agg

def form_vehicle_coalitions(vehicles, max_group_size=3):
    """Simple grouping of vehicles (placeholder for mobility-based coalition)."""
    shuffled = vehicles[:]
    random.shuffle(shuffled)
    groups = [shuffled[i:i+max_group_size] for i in range(0, len(shuffled), max_group_size)]
    print(f"[RSU] Formed {len(groups)} coalition(s): {[len(g) for g in groups]} vehicles each.")
    return groups
