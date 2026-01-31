"""
Statistical Analysis of A/B Test Results
Performs hypothesis testing and calculates business impact
"""
import pandas as pd
import numpy as np
from scipy import stats
from pathlib import Path
import logging
import yaml

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ABTestAnalyzer:
    def __init__(self, config_path='config.yaml'):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.results_dir = Path(self.config['data']['results_dir'])
        self.alpha = self.config['ab_test']['alpha']
    
    def load_results(self):
        logger.info("Loading simulation results...")
        
        control = pd.read_csv(self.results_dir / 'control_results.csv')
        treatment = pd.read_csv(self.results_dir / 'treatment_results.csv')
        
        logger.info(f"Control: {len(control)} SKUs, Treatment: {len(treatment)} SKUs")
        
        return control, treatment
    
    def analyze_primary_metrics(self, control, treatment):
        logger.info("\n" + "="*60)
        logger.info("PRIMARY METRICS ANALYSIS")
        logger.info("="*60)
        
        results = {}
        
        # 1. Fill Rate Analysis
        fill_rate_result = self.compare_means(
            control['fill_rate'], 
            treatment['fill_rate'],
            'Fill Rate (%)'
        )
        results['fill_rate'] = fill_rate_result
        
        # 2. Average Inventory Analysis
        inv_result = self.compare_means(
            control['avg_inventory'],
            treatment['avg_inventory'],
            'Average Inventory (units)'
        )
        results['avg_inventory'] = inv_result
        
        # 3. Stockout Frequency
        stockout_result = self.compare_means(
            control['stockout_count'],
            treatment['stockout_count'],
            'Stockout Count'
        )
        results['stockouts'] = stockout_result
        
        return results
    
    def compare_means(self, control, treatment, metric_name):
        """Perform t-test and calculate statistics"""
        
        # Two-sample t-test
        t_stat, p_value = stats.ttest_ind(treatment, control)
        
        # Descriptive statistics
        control_mean = control.mean()
        treatment_mean = treatment.mean()
        difference = treatment_mean - control_mean
        pct_change = (difference / control_mean) * 100
        
        # Effect size (Cohen's d)
        pooled_std = np.sqrt((control.std()**2 + treatment.std()**2) / 2)
        cohens_d = difference / pooled_std if pooled_std > 0 else 0
        
        # Confidence interval
        se = np.sqrt(control.var()/len(control) + treatment.var()/len(treatment))
        ci_lower = difference - 1.96 * se
        ci_upper = difference + 1.96 * se
        
        # Determine significance
        is_significant = p_value < self.alpha
        
        # Print results
        logger.info(f"\n{metric_name}:")
        logger.info(f"  Control Mean: {control_mean:.2f}")
        logger.info(f"  Treatment Mean: {treatment_mean:.2f}")
        logger.info(f"  Difference: {difference:.2f} ({pct_change:+.1f}%)")
        logger.info(f"  95% CI: [{ci_lower:.2f}, {ci_upper:.2f}]")
        logger.info(f"  p-value: {p_value:.4f}")
        logger.info(f"  Significant: {'YES ✓' if is_significant else 'NO ✗'}")
        logger.info(f"  Effect Size (Cohen's d): {cohens_d:.3f}")
        
        return {
            'metric': metric_name,
            'control_mean': control_mean,
            'treatment_mean': treatment_mean,
            'difference': difference,
            'pct_change': pct_change,
            'p_value': p_value,
            'is_significant': is_significant,
            'cohens_d': cohens_d,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper
        }
    
    def calculate_roi(self, control, treatment):
        logger.info("\n" + "="*60)
        logger.info("ROI CALCULATION")
        logger.info("="*60)
        
        # Inventory value reduction
        avg_unit_cost = 25  # Assume $25 average
        control_inv_value = control['avg_inventory'].sum() * avg_unit_cost
        treatment_inv_value = treatment['avg_inventory'].sum() * avg_unit_cost
        inventory_savings = control_inv_value - treatment_inv_value
        
        # Annual carrying cost savings (25% of inventory value)
        carrying_rate = self.config['simulation']['carrying_cost_rate']
        annual_carrying_savings = inventory_savings * carrying_rate
        
        # Stockout cost savings
        stockout_cost_per_incident = 150  # Lost sale + expedite fee
        control_stockouts = control['stockout_count'].sum()
        treatment_stockouts = treatment['stockout_count'].sum()
        stockout_savings = (control_stockouts - treatment_stockouts) * stockout_cost_per_incident
        
        # Implementation costs
        implementation_cost = 50000  # One-time
        annual_maintenance = 15000
        
        # Total benefit
        total_annual_benefit = annual_carrying_savings + stockout_savings - annual_maintenance
        
        # Payback period
        payback_months = (implementation_cost / (total_annual_benefit / 12)) if total_annual_benefit > 0 else np.inf
        
        # 3-year NPV (10% discount rate)
        npv = -implementation_cost
        for year in range(1, 4):
            npv += total_annual_benefit / (1.10 ** year)
        
        # ROI
        roi_year1 = ((total_annual_benefit - implementation_cost) / implementation_cost * 100)
        
        logger.info(f"\nInventory Value Reduction: ${inventory_savings:,.0f}")
        logger.info(f"Annual Carrying Cost Savings: ${annual_carrying_savings:,.0f}")
        logger.info(f"Annual Stockout Cost Savings: ${stockout_savings:,.0f}")
        logger.info(f"Implementation Cost: ${implementation_cost:,.0f}")
        logger.info(f"Annual Maintenance: ${annual_maintenance:,.0f}")
        logger.info(f"\nTotal Annual Benefit: ${total_annual_benefit:,.0f}")
        logger.info(f"Payback Period: {payback_months:.1f} months")
        logger.info(f"3-Year NPV: ${npv:,.0f}")
        logger.info(f"Year 1 ROI: {roi_year1:.1f}%")
        
        return {
            'inventory_savings': inventory_savings,
            'annual_carrying_savings': annual_carrying_savings,
            'stockout_savings': stockout_savings,
            'total_annual_benefit': total_annual_benefit,
            'payback_months': payback_months,
            'npv_3year': npv,
            'roi_year1': roi_year1
        }
    
    def generate_summary_report(self, metrics, roi):
        logger.info("\n" + "="*60)
        logger.info("EXECUTIVE SUMMARY")
        logger.info("="*60)
        
        report = []
        report.append("\nA/B TEST RESULTS: Dynamic ROP vs Fixed ROP")
        report.append("="*60)
        report.append(f"\nTest Duration: {self.config['simulation']['test_duration_days']} days")
        report.append(f"Significance Level: {self.alpha}")
        report.append("\nKEY FINDINGS:")
        report.append("-"*60)
        
        for key, result in metrics.items():
            status = "✓ SIGNIFICANT" if result['is_significant'] else "✗ Not Significant"
            report.append(f"\n{result['metric']}:")
            report.append(f"  Improvement: {result['pct_change']:+.1f}% {status}")
            report.append(f"  p-value: {result['p_value']:.4f}")
        
        report.append("\n" + "="*60)
        report.append("BUSINESS IMPACT:")
        report.append("-"*60)
        report.append(f"Annual Savings: ${roi['total_annual_benefit']:,.0f}")
        report.append(f"Payback Period: {roi['payback_months']:.1f} months")
        report.append(f"3-Year NPV: ${roi['npv_3year']:,.0f}")
        report.append(f"ROI (Year 1): {roi['roi_year1']:.1f}%")
        
        report.append("\n" + "="*60)
        report.append("RECOMMENDATION:")
        report.append("-"*60)
        
        if metrics['fill_rate']['is_significant'] and metrics['fill_rate']['pct_change'] > 0:
            report.append("\n✓ IMPLEMENT DYNAMIC REORDER POINTS")
            report.append("\nDynamic ROP shows statistically significant improvements in:")
            if metrics['fill_rate']['pct_change'] > 0:
                report.append(f"  - Fill rate: +{metrics['fill_rate']['pct_change']:.1f}%")
            if metrics['avg_inventory']['pct_change'] < 0:
                report.append(f"  - Inventory reduction: {metrics['avg_inventory']['pct_change']:.1f}%")
            if metrics['stockouts']['pct_change'] < 0:
                report.append(f"  - Stockout reduction: {metrics['stockouts']['pct_change']:.1f}%")
        else:
            report.append("\n⚠ RESULTS INCONCLUSIVE - Further testing recommended")
        
        report.append("\n" + "="*60)
        
        report_text = "\n".join(report)
        
        # Save report
        with open(self.results_dir / 'executive_summary.txt', 'w', encoding='utf-8') as f:
            f.write(report_text)
        
        logger.info(report_text)
        
        # Save detailed metrics
        metrics_df = pd.DataFrame(metrics).T
        metrics_df.to_csv(self.results_dir / 'statistical_results.csv')
        
        roi_df = pd.DataFrame([roi])
        roi_df.to_csv(self.results_dir / 'roi_analysis.csv', index=False)

def main():
    analyzer = ABTestAnalyzer()
    control, treatment = analyzer.load_results()
    
    # Statistical analysis
    metrics = analyzer.analyze_primary_metrics(control, treatment)
    
    # ROI calculation
    roi = analyzer.calculate_roi(control, treatment)
    
    # Generate summary
    analyzer.generate_summary_report(metrics, roi)
    
    logger.info("\n✅ ANALYSIS COMPLETE")
    logger.info(f"Reports saved to: {analyzer.results_dir}")

if __name__ == "__main__":
    main()