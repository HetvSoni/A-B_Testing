"""
A/B Test Simulation Engine - Run inventory simulation day-by-day
"""
import pandas as pd
import numpy as np
from pathlib import Path
import logging
import yaml
from src.rop_calculations import calculate_fixed_rop, calculate_dynamic_rop

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class InventorySimulator:
    def __init__(self, config_path='config.yaml'):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.processed_dir = Path(self.config['data']['processed_dir'])
        self.results_dir = Path(self.config['data']['results_dir'])
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        self.service_level = self.config['simulation']['service_level']
        self.test_days = self.config['simulation']['test_duration_days']
    
    def load_processed_data(self):
        logger.info("Loading processed data...")
        
        daily_demand = pd.read_csv(self.processed_dir / 'daily_demand.csv')
        daily_demand['date'] = pd.to_datetime(daily_demand['date'])
        
        sku_master = pd.read_csv(self.processed_dir / 'sku_master.csv')
        purchase_orders = pd.read_csv(self.processed_dir / 'purchase_orders.csv')
        
        logger.info(f"Loaded {len(sku_master)} SKUs, {len(daily_demand)} demand records")
        
        return daily_demand, sku_master, purchase_orders
    
    def assign_treatment_groups(self, sku_master):
        logger.info("Assigning SKUs to control/treatment groups...")
        
        np.random.seed(self.config['ab_test']['random_seed'])
        
        # Stratified randomization by ABC class
        control_skus = []
        treatment_skus = []
        
        for abc_class in ['A', 'B', 'C']:
            class_skus = sku_master[sku_master['abc_class'] == abc_class]['sku_id'].values
            n = len(class_skus)
            np.random.shuffle(class_skus)
            
            split = n // 2
            control_skus.extend(class_skus[:split])
            treatment_skus.extend(class_skus[split:])
        
        logger.info(f"Control: {len(control_skus)} SKUs, Treatment: {len(treatment_skus)} SKUs")
        
        return control_skus, treatment_skus
    
    def run_simulation(self, daily_demand, sku_master, purchase_orders):
        logger.info("Running 90-day A/B test simulation...")
        
        control_skus, treatment_skus = self.assign_treatment_groups(sku_master)
        
        # Calculate baseline stats for each SKU
        sku_stats = self.calculate_sku_statistics(daily_demand, purchase_orders)
        
        # Simulate control group (fixed ROP)
        logger.info("Simulating Control Group (Fixed ROP)...")
        control_results = self.simulate_group(
            control_skus, daily_demand, sku_stats, method='fixed'
        )
        
        # Simulate treatment group (dynamic ROP)
        logger.info("Simulating Treatment Group (Dynamic ROP)...")
        treatment_results = self.simulate_group(
            treatment_skus, daily_demand, sku_stats, method='dynamic'
        )
        
        # Save results
        control_results.to_csv(self.results_dir / 'control_results.csv', index=False)
        treatment_results.to_csv(self.results_dir / 'treatment_results.csv', index=False)
        
        logger.info("✅ Simulation complete!")
        
        return control_results, treatment_results
    
    def calculate_sku_statistics(self, daily_demand, purchase_orders):
        logger.info("Calculating SKU statistics...")
        
        stats = daily_demand.groupby('sku_id')['quantity'].agg([
            ('avg_daily_demand', 'mean'),
            ('demand_std', 'std'),
            ('total_demand', 'sum')
        ]).reset_index()
        
        # Add lead time stats
        po_stats = purchase_orders.groupby('sku_id')['lead_time_days'].agg([
            ('avg_lead_time', 'mean'),
            ('lead_time_std', 'std')
        ]).reset_index()
        
        stats = stats.merge(po_stats, on='sku_id', how='left')
        stats['avg_lead_time'] = stats['avg_lead_time'].fillna(14)
        stats['lead_time_std'] = stats['lead_time_std'].fillna(3)
        
        return stats
    
    def simulate_group(self, sku_list, daily_demand, sku_stats, method='fixed'):
        results = []
        
        for sku in sku_list:
            sku_data = sku_stats[sku_stats['sku_id'] == sku].iloc[0]
            sku_demand = daily_demand[daily_demand['sku_id'] == sku].copy()
            
            # Calculate ROP
            if method == 'fixed':
                rop_result = calculate_fixed_rop(
                    sku_data['avg_daily_demand'],
                    sku_data['avg_lead_time'],
                    sku_data['demand_std'],
                    self.service_level
                )
            else:
                recent_demand = sku_demand['quantity'].values
                recent_lt = np.array([sku_data['avg_lead_time']] * 10)
                rop_result = calculate_dynamic_rop(
                    recent_demand, recent_lt, self.service_level
                )
            
            # Simulate inventory
            sim_result = self.simulate_sku_inventory(
                sku, sku_demand, rop_result['rop'], sku_data['avg_lead_time']
            )
            
            sim_result.update({
                'sku_id': sku,
                'method': method,
                'rop': rop_result['rop'],
                'safety_stock': rop_result['safety_stock']
            })
            
            results.append(sim_result)
        
        return pd.DataFrame(results)
    
    def simulate_sku_inventory(self, sku, demand_data, rop, lead_time):
        """Simulate day-by-day inventory for one SKU"""
        
        # Initialize
        inventory = rop * 2  # Start with 2x ROP
        on_order = 0
        stockouts = 0
        total_demand = 0
        demand_met = 0
        inventory_levels = []
        
        order_qty = rop * 1.5  # Order quantity
        
        for day in range(self.test_days):
            # Daily demand (sample from historical)
            if len(demand_data) > 0:
                daily_demand = np.random.choice(demand_data['quantity'].values)
            else:
                daily_demand = 0
            
            total_demand += daily_demand
            
            # Fulfill demand
            if inventory >= daily_demand:
                inventory -= daily_demand
                demand_met += daily_demand
            else:
                demand_met += inventory
                inventory = 0
                stockouts += 1
            
            # Check if need to reorder
            if inventory <= rop and on_order == 0:
                on_order = order_qty
            
            # Receive orders (after lead time)
            if on_order > 0 and np.random.random() < (1.0 / lead_time):
                inventory += on_order
                on_order = 0
            
            inventory_levels.append(inventory)
        
        # Calculate metrics
        fill_rate = (demand_met / total_demand * 100) if total_demand > 0 else 100
        avg_inventory = np.mean(inventory_levels)
        
        return {
            'fill_rate': fill_rate,
            'avg_inventory': avg_inventory,
            'stockout_count': stockouts,
            'total_demand': total_demand,
            'demand_met': demand_met
        }

def main():
    simulator = InventorySimulator()
    daily_demand, sku_master, purchase_orders = simulator.load_processed_data()
    control_results, treatment_results = simulator.run_simulation(
        daily_demand, sku_master, purchase_orders
    )
    
    logger.info("="*60)
    logger.info("SIMULATION COMPLETE")
    logger.info("="*60)
    logger.info(f"Results saved to: {simulator.results_dir}")

if __name__ == "__main__":
    main()