"""Data preprocessing for Amazon FBA analysis"""
import pandas as pd
import numpy as np
from pathlib import Path
import logging
import yaml

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AmazonFBAPreprocessor:
    def __init__(self, config_path='config.yaml'):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        self.raw_dir = Path(self.config['data']['raw_dir'])
        self.processed_dir = Path(self.config['data']['processed_dir'])
        self.processed_dir.mkdir(parents=True, exist_ok=True)
    
    def load_data(self):
        logger.info("Loading raw data...")
        csv_files = list(self.raw_dir.glob('*.csv'))
        main_file = max(csv_files, key=lambda f: f.stat().st_size)
        df = pd.read_csv(main_file)
        logger.info(f"Loaded {len(df):,} records")
        return df
    
    def clean_data(self, df):
        logger.info("Cleaning data...")
        df = df.drop_duplicates()
        df = df.dropna()
        return df
    
    def add_amazon_features(self, df):
        logger.info("Adding Amazon FBA features...")
        np.random.seed(42)
        n = len(df)
        
        fc_options = ['PHX3', 'BFI4', 'MDW2', 'EWR4', 'DFW6']
        df['fulfillment_center'] = np.random.choice(fc_options, n)
        df['prime_eligible'] = np.random.choice([True, False], n, p=[0.9, 0.1])
        df['storage_type'] = np.random.choice(['Standard', 'Oversized'], n, p=[0.8, 0.2])
        
        return df
    
    def create_abc_classification(self, df, sku_col, qty_col, cost_col):
        logger.info("Creating ABC classification...")
        revenue = df.groupby(sku_col).agg({
            qty_col: 'sum',
            cost_col: 'mean'
        }).reset_index()
        
        revenue['total_revenue'] = revenue[qty_col] * revenue[cost_col]
        revenue = revenue.sort_values('total_revenue', ascending=False)
        revenue['cum_pct'] = revenue['total_revenue'].cumsum() / revenue['total_revenue'].sum()
        
        revenue['abc_class'] = 'C'
        revenue.loc[revenue['cum_pct'] <= 0.80, 'abc_class'] = 'A'
        revenue.loc[(revenue['cum_pct'] > 0.80) & (revenue['cum_pct'] <= 0.95), 'abc_class'] = 'B'
        
        df = df.merge(revenue[[sku_col, 'abc_class']], on=sku_col, how='left')
        return df
    
    def create_output_tables(self, df):
        logger.info("Creating output tables...")
        
        # Auto-detect columns
        date_cols = [c for c in df.columns if 'date' in c.lower()]
        sku_cols = [c for c in df.columns if any(x in c.lower() for x in ['product', 'sku', 'item'])]
        qty_cols = [c for c in df.columns if any(x in c.lower() for x in ['quantity', 'sales', 'demand'])]
        cost_cols = [c for c in df.columns if any(x in c.lower() for x in ['price', 'cost'])]
        
        date_col = date_cols[0] if date_cols else df.columns[0]
        sku_col = sku_cols[0] if sku_cols else df.columns[1]
        qty_col = qty_cols[0] if qty_cols else df.columns[2]
        cost_col = cost_cols[0] if cost_cols else df.columns[3]
        
        # Daily demand
        daily_demand = df.groupby([date_col, sku_col])[qty_col].sum().reset_index()
        daily_demand.columns = ['date', 'sku_id', 'quantity']
        daily_demand.to_csv(self.processed_dir / 'daily_demand.csv', index=False)
        
        # SKU master
        sku_master = df.groupby(sku_col).agg({
            cost_col: 'mean',
            'abc_class': 'first',
            'fulfillment_center': 'first',
            'storage_type': 'first'
        }).reset_index()
        sku_master.columns = ['sku_id', 'unit_cost', 'abc_class', 'fc', 'storage']
        sku_master.to_csv(self.processed_dir / 'sku_master.csv', index=False)
        
        # Purchase orders
        po = self.generate_purchase_orders(df, sku_col)
        po.to_csv(self.processed_dir / 'purchase_orders.csv', index=False)
        
        logger.info("✅ Created all output files")
        return {'daily_demand': daily_demand, 'sku_master': sku_master, 'po': po}
    
    def generate_purchase_orders(self, df, sku_col):
        logger.info("Generating purchase orders...")
        np.random.seed(42)
        skus = df[sku_col].unique()
        pos = []
        
        for sku in skus:
            for i in range(20):
                order_date = pd.Timestamp.now() - pd.Timedelta(days=np.random.randint(0, 365))
                lead_time = max(5, int(np.random.normal(14, 3)))
                receipt_date = order_date + pd.Timedelta(days=lead_time)
                
                pos.append({
                    'po_id': f'PO_{sku}_{i:04d}',
                    'sku_id': sku,
                    'order_date': order_date,
                    'receipt_date': receipt_date,
                    'lead_time_days': lead_time,
                    'quantity': 100
                })
        
        return pd.DataFrame(pos)

def main():
    processor = AmazonFBAPreprocessor()
    df = processor.load_data()
    df = processor.clean_data(df)
    df = processor.add_amazon_features(df)
    
    # Detect columns and create ABC
    sku_col = [c for c in df.columns if 'product' in c.lower() or 'sku' in c.lower()][0]
    qty_col = [c for c in df.columns if 'quantity' in c.lower() or 'sales' in c.lower()][0]
    cost_col = [c for c in df.columns if 'price' in c.lower() or 'cost' in c.lower()][0]
    
    df = processor.create_abc_classification(df, sku_col, qty_col, cost_col)
    tables = processor.create_output_tables(df)
    
    logger.info("✅ PREPROCESSING COMPLETE")

if __name__ == "__main__":
    main()