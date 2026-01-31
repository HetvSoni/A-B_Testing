"""
Reorder Point Calculations - Fixed vs Dynamic
"""
import numpy as np
import pandas as pd
from scipy import stats

def calculate_fixed_rop(avg_daily_demand, avg_lead_time, demand_std, service_level=0.95):
    """
    Calculate traditional fixed reorder point
    
    ROP = (Avg Daily Demand × Avg Lead Time) + Safety Stock
    """
    z_score = stats.norm.ppf(service_level)
    safety_stock = z_score * demand_std * np.sqrt(avg_lead_time)
    reorder_point = (avg_daily_demand * avg_lead_time) + safety_stock
    
    return {
        'rop': reorder_point,
        'safety_stock': safety_stock,
        'method': 'fixed'
    }

def calculate_dynamic_rop(recent_demand, recent_lead_times, service_level=0.95, 
                         weights={'30d': 0.5, '60d': 0.3, '90d': 0.2}):
    """
    Calculate dynamic reorder point using weighted moving average
    
    ROP = (WMA Demand × Forecasted Lead Time) + Dynamic Safety Stock
    """
    # Weighted moving average demand
    if len(recent_demand) >= 90:
        demand_30d = recent_demand[-30:].mean()
        demand_60d = recent_demand[-60:].mean()
        demand_90d = recent_demand[-90:].mean()
        wma_demand = (weights['30d'] * demand_30d + 
                     weights['60d'] * demand_60d + 
                     weights['90d'] * demand_90d)
    else:
        wma_demand = recent_demand.mean()
    
    # Forecasted lead time (recent average)
    if len(recent_lead_times) > 0:
        forecasted_lt = recent_lead_times[-10:].mean()
    else:
        forecasted_lt = 14  # Default
    
    # Dynamic safety stock (based on recent volatility)
    recent_std = recent_demand[-30:].std() if len(recent_demand) >= 30 else recent_demand.std()
    z_score = stats.norm.ppf(service_level)
    safety_stock = z_score * recent_std * np.sqrt(forecasted_lt)
    
    reorder_point = (wma_demand * forecasted_lt) + safety_stock
    
    return {
        'rop': reorder_point,
        'safety_stock': safety_stock,
        'wma_demand': wma_demand,
        'forecasted_lt': forecasted_lt,
        'method': 'dynamic'
    }

def calculate_eoq(annual_demand, order_cost=50, holding_cost_rate=0.25, unit_cost=10):
    """
    Calculate Economic Order Quantity
    """
    holding_cost = unit_cost * holding_cost_rate
    eoq = np.sqrt((2 * annual_demand * order_cost) / holding_cost)
    return eoq