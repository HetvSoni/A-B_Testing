"""
Visualization Module - Create charts for A/B test results
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import logging
import yaml

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10

class ABTestVisualizer:
    def __init__(self, config_path='config.yaml'):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.results_dir = Path(self.config['data']['results_dir'])
        self.figures_dir = Path('results/figures')
        self.figures_dir.mkdir(parents=True, exist_ok=True)
    
    def load_data(self):
        logger.info("Loading results for visualization...")
        
        control = pd.read_csv(self.results_dir / 'control_results.csv')
        treatment = pd.read_csv(self.results_dir / 'treatment_results.csv')
        metrics = pd.read_csv(self.results_dir / 'statistical_results.csv', index_col=0)
        
        return control, treatment, metrics
    
    def create_comparison_dashboard(self, control, treatment, metrics):
        """Create executive dashboard comparing control vs treatment"""
        logger.info("Creating comparison dashboard...")
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Dynamic ROP vs Fixed ROP - A/B Test Results', 
                     fontsize=16, fontweight='bold')
        
        # 1. Fill Rate Comparison
        ax1 = axes[0, 0]
        data = pd.DataFrame({
            'Group': ['Fixed ROP\n(Control)', 'Dynamic ROP\n(Treatment)'],
            'Fill Rate (%)': [control['fill_rate'].mean(), treatment['fill_rate'].mean()]
        })
        bars = ax1.bar(data['Group'], data['Fill Rate (%)'], 
                       color=['#ff6b6b', '#4ecdc4'], alpha=0.8, edgecolor='black')
        ax1.set_ylabel('Fill Rate (%)', fontweight='bold')
        ax1.set_title('Fill Rate Comparison', fontweight='bold')
        ax1.set_ylim(0, 100)
        ax1.axhline(95, color='green', linestyle='--', label='Target: 95%')
        ax1.legend()
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.1f}%', ha='center', va='bottom', fontweight='bold')
        
        # 2. Average Inventory Comparison
        ax2 = axes[0, 1]
        data = pd.DataFrame({
            'Group': ['Fixed ROP\n(Control)', 'Dynamic ROP\n(Treatment)'],
            'Avg Inventory': [control['avg_inventory'].mean(), treatment['avg_inventory'].mean()]
        })
        bars = ax2.bar(data['Group'], data['Avg Inventory'], 
                       color=['#ff6b6b', '#4ecdc4'], alpha=0.8, edgecolor='black')
        ax2.set_ylabel('Average Inventory (units)', fontweight='bold')
        ax2.set_title('Average Inventory Comparison', fontweight='bold')
        
        for bar in bars:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.0f}', ha='center', va='bottom', fontweight='bold')
        
        # 3. Stockout Frequency
        ax3 = axes[1, 0]
        data = pd.DataFrame({
            'Group': ['Fixed ROP\n(Control)', 'Dynamic ROP\n(Treatment)'],
            'Stockouts': [control['stockout_count'].mean(), treatment['stockout_count'].mean()]
        })
        bars = ax3.bar(data['Group'], data['Stockouts'], 
                       color=['#ff6b6b', '#4ecdc4'], alpha=0.8, edgecolor='black')
        ax3.set_ylabel('Average Stockout Count', fontweight='bold')
        ax3.set_title('Stockout Frequency Comparison', fontweight='bold')
        
        for bar in bars:
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.1f}', ha='center', va='bottom', fontweight='bold')
        
        # 4. Statistical Significance Summary
        ax4 = axes[1, 1]
        ax4.axis('off')
        
        summary_text = "Statistical Significance Summary\n\n"
        summary_text += f"Fill Rate:\n"
        summary_text += f"  Improvement: {metrics.loc['fill_rate', 'pct_change']:+.1f}%\n"
        summary_text += f"  p-value: {metrics.loc['fill_rate', 'p_value']:.4f}\n"
        summary_text += f"  Status: {'SIGNIFICANT' if metrics.loc['fill_rate', 'is_significant'] else 'Not Significant'}\n\n"
        
        summary_text += f"Stockouts:\n"
        summary_text += f"  Reduction: {metrics.loc['stockouts', 'pct_change']:.1f}%\n"
        summary_text += f"  p-value: {metrics.loc['stockouts', 'p_value']:.4f}\n"
        summary_text += f"  Status: {'SIGNIFICANT' if metrics.loc['stockouts', 'is_significant'] else 'Not Significant'}\n\n"
        
        summary_text += f"Sample Size:\n"
        summary_text += f"  Control: {len(control)} SKUs\n"
        summary_text += f"  Treatment: {len(treatment)} SKUs\n"
        
        ax4.text(0.1, 0.9, summary_text, transform=ax4.transAxes,
                fontsize=11, verticalalignment='top', family='monospace',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
        
        plt.tight_layout()
        plt.savefig(self.figures_dir / 'comparison_dashboard.png', dpi=300, bbox_inches='tight')
        logger.info(f"Saved: comparison_dashboard.png")
        plt.close()
    
    def create_distribution_plots(self, control, treatment):
        """Create distribution plots for key metrics"""
        logger.info("Creating distribution plots...")
        
        fig, axes = plt.subplots(1, 3, figsize=(16, 5))
        fig.suptitle('Distribution Comparison: Control vs Treatment', 
                     fontsize=14, fontweight='bold')
        
        metrics_to_plot = [
            ('fill_rate', 'Fill Rate (%)', axes[0]),
            ('avg_inventory', 'Average Inventory (units)', axes[1]),
            ('stockout_count', 'Stockout Count', axes[2])
        ]
        
        for metric, title, ax in metrics_to_plot:
            # Plot distributions
            ax.hist(control[metric], bins=15, alpha=0.6, label='Fixed ROP', 
                   color='#ff6b6b', edgecolor='black')
            ax.hist(treatment[metric], bins=15, alpha=0.6, label='Dynamic ROP', 
                   color='#4ecdc4', edgecolor='black')
            
            # Add mean lines
            ax.axvline(control[metric].mean(), color='#ff6b6b', 
                      linestyle='--', linewidth=2, label='Control Mean')
            ax.axvline(treatment[metric].mean(), color='#4ecdc4', 
                      linestyle='--', linewidth=2, label='Treatment Mean')
            
            ax.set_xlabel(title, fontweight='bold')
            ax.set_ylabel('Frequency', fontweight='bold')
            ax.set_title(title, fontweight='bold')
            ax.legend()
            ax.grid(alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.figures_dir / 'distribution_plots.png', dpi=300, bbox_inches='tight')
        logger.info(f"Saved: distribution_plots.png")
        plt.close()
    
    def create_scatter_analysis(self, control, treatment):
        """Create scatter plots showing relationships"""
        logger.info("Creating scatter analysis...")
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        fig.suptitle('Fill Rate vs Inventory Analysis', fontsize=14, fontweight='bold')
        
        # Control group
        axes[0].scatter(control['avg_inventory'], control['fill_rate'], 
                       s=100, alpha=0.6, color='#ff6b6b', edgecolor='black')
        axes[0].set_xlabel('Average Inventory (units)', fontweight='bold')
        axes[0].set_ylabel('Fill Rate (%)', fontweight='bold')
        axes[0].set_title('Fixed ROP (Control)', fontweight='bold')
        axes[0].grid(alpha=0.3)
        axes[0].axhline(95, color='green', linestyle='--', alpha=0.5, label='Target Fill Rate')
        axes[0].legend()
        
        # Treatment group
        axes[1].scatter(treatment['avg_inventory'], treatment['fill_rate'], 
                       s=100, alpha=0.6, color='#4ecdc4', edgecolor='black')
        axes[1].set_xlabel('Average Inventory (units)', fontweight='bold')
        axes[1].set_ylabel('Fill Rate (%)', fontweight='bold')
        axes[1].set_title('Dynamic ROP (Treatment)', fontweight='bold')
        axes[1].grid(alpha=0.3)
        axes[1].axhline(95, color='green', linestyle='--', alpha=0.5, label='Target Fill Rate')
        axes[1].legend()
        
        plt.tight_layout()
        plt.savefig(self.figures_dir / 'scatter_analysis.png', dpi=300, bbox_inches='tight')
        logger.info(f"Saved: scatter_analysis.png")
        plt.close()
    
    def create_metrics_summary_table(self, metrics):
        """Create a visual summary table"""
        logger.info("Creating metrics summary table...")
        
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.axis('tight')
        ax.axis('off')
        
        # Prepare table data
        table_data = []
        for idx, row in metrics.iterrows():
            sig = "YES" if row['is_significant'] else "NO"
            table_data.append([
                row['metric'],
                f"{row['control_mean']:.2f}",
                f"{row['treatment_mean']:.2f}",
                f"{row['pct_change']:+.1f}%",
                f"{row['p_value']:.4f}",
                sig
            ])
        
        # Create table
        table = ax.table(cellText=table_data,
                        colLabels=['Metric', 'Control', 'Treatment', 'Change', 'p-value', 'Significant'],
                        cellLoc='center',
                        loc='center',
                        colWidths=[0.25, 0.15, 0.15, 0.15, 0.15, 0.15])
        
        table.auto_set_font_size(False)
        table.set_fontsize(11)
        table.scale(1, 2)
        
        # Color header
        for i in range(6):
            table[(0, i)].set_facecolor('#4ecdc4')
            table[(0, i)].set_text_props(weight='bold', color='white')
        
        # Color significant rows
        for i, row in enumerate(metrics.iterrows(), 1):
            if row[1]['is_significant']:
                for j in range(6):
                    table[(i, j)].set_facecolor('#d4edda')
        
        plt.title('Statistical Analysis Summary', fontsize=14, fontweight='bold', pad=20)
        plt.savefig(self.figures_dir / 'metrics_summary_table.png', dpi=300, bbox_inches='tight')
        logger.info(f"Saved: metrics_summary_table.png")
        plt.close()

def main():
    visualizer = ABTestVisualizer()
    control, treatment, metrics = visualizer.load_data()
    
    # Create all visualizations
    visualizer.create_comparison_dashboard(control, treatment, metrics)
    visualizer.create_distribution_plots(control, treatment)
    visualizer.create_scatter_analysis(control, treatment)
    visualizer.create_metrics_summary_table(metrics)
    
    logger.info("\n" + "="*60)
    logger.info("✅ ALL VISUALIZATIONS CREATED")
    logger.info("="*60)
    logger.info(f"Location: {visualizer.figures_dir}")
    logger.info("\nGenerated files:")
    logger.info("  1. comparison_dashboard.png")
    logger.info("  2. distribution_plots.png")
    logger.info("  3. scatter_analysis.png")
    logger.info("  4. metrics_summary_table.png")

if __name__ == "__main__":
    main()