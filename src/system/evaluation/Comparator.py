# src/system/evaluation/Comparator.py

import json
import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Any, Optional, Tuple, Union
from collections import defaultdict
from scipy.interpolate import interp1d
from datetime import datetime

from config import Config

class Comparator:
    """
    Tool for comparing object detection metrics across multiple evaluation runs.
    Allows handpicking which detection sources from which runs to compare with custom styling.
    """
    
    def __init__(self, eval_base_folder: str = None, config: Config = None):
        """
        Initialize comparator
        
        Args:
            eval_base_folder: Base folder containing timestamped evaluation runs.
        """
        if eval_base_folder is None:
            eval_base_folder = self._auto_detect_eval_folder()
        
        self.eval_base_folder = eval_base_folder
        self.available_runs = {}
        self.loaded_metrics = {}
        self.config = config
        
        # Default color mapping for consistency
        self.default_color_map = {
            'depth': "#9F33D1",     # Purple
            'midas': "#407FBE",     # Blue  
            'pro': "#349649",       # Green
            'vggt': "#FCEA4C",      # Yellow
            'color': "#F14B4B"      # Red
        }
        
        # Line styles for differentiation
        self.line_styles = ['-', '--', '-.', ':', '-', '--', '-.', ':']
        
        print(f"Using evaluation folder: {self.eval_base_folder}")
        self.discover_runs()
    
    def _auto_detect_eval_folder(self) -> str:
        """Auto-detect evaluation folder based on standard project structure"""
        # Get the root directory of this script
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(os.path.dirname(script_dir))
        
        # Standard evaluation output path
        eval_run_type = self.config.eval_run_type 
        eval_output_folder = os.path.join(project_root, "testing/results", eval_run_type)
        
        if not os.path.exists(eval_output_folder):
            print(f"Warning: Auto-detected eval folder doesn't exist: {eval_output_folder}")
            print("You may need to specify the path manually")
        
        return eval_output_folder
    
    def get_results_folder(self) -> str:
        """Get the base results folder for saving comparison outputs"""
        # Save comparisons in the same results structure
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(os.path.dirname(script_dir))
        eval_run_type = "object_detection"
        results_base = os.path.join(project_root, "testing/results", eval_run_type)
        
        # Create a comparisons subfolder
        comparisons_folder = os.path.join(results_base, "comparisons")
        os.makedirs(comparisons_folder, exist_ok=True)
        
        return comparisons_folder
    
    def discover_runs(self):
        """Discover available evaluation runs"""
        if not os.path.exists(self.eval_base_folder):
            print(f"Evaluation folder not found: {self.eval_base_folder}")
            return
            
        pattern = os.path.join(self.eval_base_folder, "*", "evaluation_metrics.json")
        metric_files = glob.glob(pattern)
        
        print(f"Searching for metrics in: {pattern}")
        
        for metric_file in metric_files:
            run_dir = os.path.dirname(metric_file)
            run_name = os.path.basename(run_dir)
            
            # Try to load and check what sources are available
            try:
                with open(metric_file, 'r') as f:
                    metrics = json.load(f)
                
                # Extract available detection sources
                sources = []
                for key in metrics.keys():
                    if key.endswith('_detections') or key in ['depth_detections', 'midas_detections', 'pro_detections', 'vggt_detections', 'color_detections']:
                        sources.append(key)
                
                self.available_runs[run_name] = {
                    'path': metric_file,
                    'sources': sources,
                    'metrics': None  # Will load on demand
                }
                
                print(f"Found run: {run_name} with sources: {sources}")
                
            except Exception as e:
                print(f"Failed to load {metric_file}: {e}")
        
        if not self.available_runs:
            print("No evaluation runs found!")
    
    def load_run_metrics(self, run_name: str) -> Optional[Dict[str, Any]]:
        """Load metrics for a specific run"""
        if run_name not in self.available_runs:
            print(f"Run {run_name} not found")
            return None
        
        if self.available_runs[run_name]['metrics'] is None:
            try:
                with open(self.available_runs[run_name]['path'], 'r') as f:
                    self.available_runs[run_name]['metrics'] = json.load(f)
            except Exception as e:
                print(f"Failed to load metrics for {run_name}: {e}")
                return None
        
        return self.available_runs[run_name]['metrics']
    
    def list_available_runs(self):
        """Print all available runs and their sources"""
        print("\nAvailable Evaluation Runs:")
        print("=" * 50)
        
        if not self.available_runs:
            print("No runs found!")
            return
        
        for run_name, run_info in self.available_runs.items():
            print(f"\nRun: {run_name}")
            print(f"  Sources: {', '.join(run_info['sources'])}")
    
    def create_comparison_dataset(self, selections: List[Union[Tuple[str, str], Tuple[str, str, str], Tuple[str, str, str, str]]]) -> Dict[str, Any]:
        """
        Create a comparison dataset from selected run/source combinations with custom styling
        
        Args:
            selections: List of tuples in format:
                       (run_name, source_name) or
                       (run_name, source_name, custom_name) or  
                       (run_name, source_name, custom_name, line_style)
            
        Returns:
            Dictionary structured like original metrics but with selected combinations
        """
        comparison_data = {}
        
        for i, selection in enumerate(selections):
            # Parse selection tuple
            if len(selection) == 2:
                run_name, source_name = selection
                custom_name = f"{source_name}_{run_name}"
                line_style = self.line_styles[i % len(self.line_styles)]
            elif len(selection) == 3:
                run_name, source_name, custom_name = selection
                line_style = self.line_styles[i % len(self.line_styles)]
            elif len(selection) == 4:
                run_name, source_name, custom_name, line_style = selection
            else:
                print(f"Invalid selection format: {selection}")
                continue
            
            # Load metrics for this run
            metrics = self.load_run_metrics(run_name)
            if metrics is None or source_name not in metrics:
                print(f"Warning: {source_name} not found in run {run_name}")
                continue
            
            # Create entry with custom styling
            comparison_data[custom_name] = metrics[source_name].copy()
            
            # Add metadata including styling info
            comparison_data[custom_name]['_meta'] = {
                'run_name': run_name,
                'source_name': source_name,
                'custom_name': custom_name,
                'line_style': line_style,
                'original_source': source_name
            }
        
        return comparison_data
    
    def _get_color_for_source(self, source_name: str, index: int) -> str:
        """Get color for a source, with fallback to indexed colors"""
        source_key = source_name.replace('_detections', '')
        if source_key in self.default_color_map:
            return self.default_color_map[source_key]
        else:
            # Generate distinct colors for custom sources
            colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FECA57', '#FF9FF3', '#54A0FF', '#5F27CD']
            return colors[index % len(colors)]
    
    def plot_ap_comparison(self, comparison_data: Dict[str, Any], output_path: str):
        """Plot Average Precision comparison across selected combinations"""
        combo_names = []
        aps = []
        colors = []
        
        for i, (combo_key, data) in enumerate(comparison_data.items()):
            if 'overall_metrics' in data:
                meta = data.get('_meta', {})
                display_name = meta.get('custom_name', combo_key)
                source_name = meta.get('source_name', '')
                
                combo_names.append(display_name)
                aps.append(data['overall_metrics'].get('average_precision', 0))
                colors.append(self._get_color_for_source(source_name, i))
        
        if not combo_names:
            print("No valid data for AP comparison")
            return
        
        plt.figure(figsize=(max(12, len(combo_names) * 1.5), 8))
        bars = plt.bar(combo_names, aps, color=colors)
        
        plt.ylabel('Average Precision', fontsize=12)
        plt.title('Object Detection Performance Comparison', fontsize=14)
        plt.xticks(rotation=45, ha='right')
        plt.ylim(0, 1)
        
        # Add value labels on bars
        for bar, ap in zip(bars, aps):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{ap:.3f}', ha='center', va='bottom', fontweight='bold')
        
        plt.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved AP comparison to {output_path}")
    
    def plot_pr_curves_comparison(self, comparison_data: Dict[str, Any], output_path: str):
        """Plot Precision-Recall curves comparison with custom styling"""
        plt.figure(figsize=(12, 8))
        
        for i, (combo_key, data) in enumerate(comparison_data.items()):
            if 'overall_metrics' in data:
                overall = data['overall_metrics']
                if 'precision_recall_curve' in overall:
                    pr_data = overall['precision_recall_curve']
                    precisions = pr_data.get('precisions', [])
                    recalls = pr_data.get('recalls', [])
                    
                    if len(precisions) > 1 and len(recalls) > 1:
                        # Sort by recall for proper curve
                        sorted_pairs = sorted(zip(recalls, precisions))
                        sorted_recalls, sorted_precisions = zip(*sorted_pairs)
                        
                        # Convert to numpy arrays
                        recalls_np = np.array(sorted_recalls)
                        precisions_np = np.array(sorted_precisions)
                        
                        # Smooth curve if enough points
                        if len(recalls_np) > 10:
                            # Remove duplicates for interpolation
                            unique_indices = np.unique(recalls_np, return_index=True)[1]
                            recalls_unique = recalls_np[unique_indices]
                            precisions_unique = precisions_np[unique_indices]
                            
                            if len(recalls_unique) > 3:
                                f = interp1d(recalls_unique, precisions_unique, 
                                           kind='cubic', fill_value='extrapolate')
                                recalls_smooth = np.linspace(recalls_unique.min(), 
                                                           recalls_unique.max(), 100)
                                precisions_smooth = f(recalls_smooth)
                                precisions_smooth = np.clip(precisions_smooth, 0, 1)
                            else:
                                recalls_smooth = recalls_unique
                                precisions_smooth = precisions_unique
                        else:
                            recalls_smooth = recalls_np
                            precisions_smooth = precisions_np
                        
                        # Get styling info
                        meta = data.get('_meta', {})
                        display_name = meta.get('custom_name', combo_key)
                        source_name = meta.get('source_name', '')
                        line_style = meta.get('line_style', '-')
                        ap = overall.get('average_precision', 0)
                        
                        color = self._get_color_for_source(source_name, i)
                        label = f"{display_name} - AP={ap:.3f}"
                        
                        plt.plot(recalls_smooth, precisions_smooth,
                                color=color, linestyle=line_style, 
                                label=label, linewidth=2.5)
        
        plt.xlabel('Recall', fontsize=12)
        plt.ylabel('Precision', fontsize=12)
        plt.title('Precision-Recall Curves Comparison', fontsize=14)
        plt.legend(loc='lower left', fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.xlim([0, 1])
        plt.ylim([0, 1.05])
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved PR curves comparison to {output_path}")
    
    def plot_threshold_analysis_comparison(self, comparison_data: Dict[str, Any], output_path: str):
        """Plot threshold analysis comparison with custom styling"""
        n_plots = len(comparison_data)
        if n_plots == 0:
            return
        
        cols = min(3, n_plots)
        rows = (n_plots + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(5*cols, 4*rows))
        if n_plots == 1:
            axes = [axes]
        elif rows == 1:
            axes = axes.reshape(1, -1)
        
        axes_flat = axes.ravel() if isinstance(axes, np.ndarray) else [axes]
        
        for idx, (combo_key, data) in enumerate(comparison_data.items()):
            if idx >= len(axes_flat):
                break
                
            ax = axes_flat[idx]
            
            if 'threshold_analysis' in data:
                thresh_data = data['threshold_analysis']
                thresholds = []
                precisions = []
                recalls = []
                f1_scores = []
                
                for thresh_key in sorted(thresh_data.keys()):
                    thresh_info = thresh_data[thresh_key]
                    thresholds.append(thresh_info['threshold'])
                    precisions.append(thresh_info['precision'])
                    recalls.append(thresh_info['recall'])
                    f1_scores.append(thresh_info['f1_score'])
                
                # Get styling info
                meta = data.get('_meta', {})
                line_style = meta.get('line_style', '-')
                
                ax.plot(thresholds, precisions, 'b-', label='Precision', 
                       linewidth=2, linestyle=line_style)
                ax.plot(thresholds, recalls, 'r-', label='Recall', 
                       linewidth=2, linestyle=line_style)
                ax.plot(thresholds, f1_scores, 'g-', label='F1 Score', 
                       linewidth=2, linestyle=line_style)
                
                # Mark optimal threshold if available
                optimal_thresh = data.get('optimal_threshold', 0.5)
                ax.axvline(x=optimal_thresh, color='black', linestyle='--', 
                          alpha=0.7, label=f'Optimal ({optimal_thresh:.2f})')
                
                # Use custom name for title
                display_name = meta.get('custom_name', combo_key)
                
                ax.set_xlabel('Confidence Threshold')
                ax.set_ylabel('Score')
                ax.set_title(display_name)
                ax.legend()
                ax.grid(True, alpha=0.3)
                ax.set_xlim(0, 1)
                ax.set_ylim(0, 1.1)
        
        # Hide unused subplots
        for idx in range(len(comparison_data), len(axes_flat)):
            axes_flat[idx].axis('off')
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved threshold analysis comparison to {output_path}")
    
    def plot_metrics_summary_comparison(self, comparison_data: Dict[str, Any], output_path: str):
        """Plot summary metrics comparison with custom styling"""
        combo_names = []
        precisions = []
        recalls = []
        f1_scores = []
        aps = []
        
        for combo_key, data in comparison_data.items():
            if 'overall_metrics' in data:
                overall = data['overall_metrics']
                meta = data.get('_meta', {})
                display_name = meta.get('custom_name', combo_key)
                
                combo_names.append(display_name)
                precisions.append(overall.get('precision', 0))
                recalls.append(overall.get('recall', 0))
                f1_scores.append(overall.get('f1_score', 0))
                aps.append(overall.get('average_precision', 0))
        
        if not combo_names:
            print("No valid data for metrics summary")
            return
        
        x = np.arange(len(combo_names))
        width = 0.2
        
        fig, ax = plt.subplots(figsize=(max(15, len(combo_names) * 2), 8))
        
        bars1 = ax.bar(x - 1.5*width, precisions, width, label='Precision', color='skyblue')
        bars2 = ax.bar(x - 0.5*width, recalls, width, label='Recall', color='lightcoral')
        bars3 = ax.bar(x + 0.5*width, f1_scores, width, label='F1 Score', color='lightgreen')
        bars4 = ax.bar(x + 1.5*width, aps, width, label='Average Precision', color='gold')
        
        ax.set_xlabel('Detection Method')
        ax.set_ylabel('Score')
        ax.set_title('Detection Performance Metrics Comparison')
        ax.set_xticks(x)
        ax.set_xticklabels(combo_names, rotation=45, ha='right')
        ax.legend()
        ax.set_ylim(0, 1.1)
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bars
        def autolabel(bars):
            for bar in bars:
                height = bar.get_height()
                ax.annotate(f'{height:.2f}',
                           xy=(bar.get_x() + bar.get_width() / 2, height),
                           xytext=(0, 3),
                           textcoords="offset points",
                           ha='center', va='bottom',
                           fontsize=8)
        
        autolabel(bars1)
        autolabel(bars2)
        autolabel(bars3)
        autolabel(bars4)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved metrics summary to {output_path}")
    
    def generate_comparison_report(self, selections: List[Union[Tuple[str, str], Tuple[str, str, str], Tuple[str, str, str, str]]], 
                                 output_dir: str = None, report_name: str = None):
        """
        Generate a complete comparison report with all plots
        
        Args:
            selections: List of selection tuples with optional custom names and line styles
            output_dir: Directory to save comparison results (default: auto-detected results folder)
            report_name: Optional name for the report (default: timestamp)
        """
        if output_dir is None:
            output_dir = self.get_results_folder()
        
        if report_name is None:
            report_name = f"comparison_{datetime.now().strftime('%m%d_%H%M%S')}"
        
        report_dir = os.path.join(output_dir, report_name)
        os.makedirs(report_dir, exist_ok=True)
        
        print(f"\nGenerating comparison report: {report_name}")
        print(f"Output directory: {report_dir}")
        print(f"Comparing {len(selections)} combinations:")
        for selection in selections:
            if len(selection) >= 3:
                print(f"  - {selection[2]} ({selection[0]}/{selection[1]})")
            else:
                print(f"  - {selection[0]}/{selection[1]}")
        
        # Create comparison dataset
        comparison_data = self.create_comparison_dataset(selections)
        
        if not comparison_data:
            print("No valid comparison data found!")
            return
        
        # Generate all plots
        self.plot_ap_comparison(comparison_data, 
                               os.path.join(report_dir, "ap_comparison.png"))
        
        self.plot_pr_curves_comparison(comparison_data, 
                                     os.path.join(report_dir, "pr_curves_comparison.png"))
        
        self.plot_threshold_analysis_comparison(comparison_data, 
                                              os.path.join(report_dir, "threshold_analysis_comparison.png"))
        
        self.plot_metrics_summary_comparison(comparison_data, 
                                           os.path.join(report_dir, "metrics_summary_comparison.png"))
        
        # Save comparison data as JSON
        comparison_json_path = os.path.join(report_dir, "comparison_data.json")
        with open(comparison_json_path, 'w') as f:
            json.dump(comparison_data, f, indent=2)
        
        # Generate summary text report
        summary_path = os.path.join(report_dir, "comparison_summary.txt")
        self._generate_text_summary(comparison_data, summary_path, selections)
        
        print(f"\nComparison report generated successfully!")
        print(f"Files saved in: {report_dir}")
        
        return report_dir
    
    def _generate_text_summary(self, comparison_data: Dict[str, Any], 
                             output_path: str, selections: List[Tuple]):
        """Generate a text summary of the comparison"""
        with open(output_path, 'w') as f:
            f.write("DETECTION PERFORMANCE COMPARISON REPORT\n")
            f.write("=" * 50 + "\n\n")
            
            f.write("COMPARED COMBINATIONS:\n")
            f.write("-" * 25 + "\n")
            for selection in selections:
                if len(selection) >= 3:
                    custom_name = selection[2]
                    run_name = selection[0]
                    source_name = selection[1]
                    f.write(f"  {custom_name}: {source_name} from {run_name}")
                    if len(selection) >= 4:
                        f.write(f" (style: {selection[3]})")
                    f.write("\n")
                else:
                    run_name, source_name = selection[:2]
                    f.write(f"  {source_name} from run {run_name}\n")
            f.write("\n")
            
            f.write("PERFORMANCE SUMMARY:\n")
            f.write("-" * 20 + "\n")
            
            # Sort by AP for ranking
            ranked_data = []
            for combo_key, data in comparison_data.items():
                if 'overall_metrics' in data:
                    overall = data['overall_metrics']
                    meta = data.get('_meta', {})
                    ap = overall.get('average_precision', 0)
                    display_name = meta.get('custom_name', combo_key)
                    ranked_data.append((ap, display_name, data, meta))
            
            ranked_data.sort(reverse=True, key=lambda x: x[0])
            
            for rank, (ap, display_name, data, meta) in enumerate(ranked_data, 1):
                overall = data['overall_metrics']
                
                f.write(f"\n{rank}. {display_name}:\n")
                f.write(f"   Average Precision: {overall.get('average_precision', 0):.3f}\n")
                f.write(f"   Precision:         {overall.get('precision', 0):.3f}\n")
                f.write(f"   Recall:            {overall.get('recall', 0):.3f}\n")
                f.write(f"   F1 Score:          {overall.get('f1_score', 0):.3f}\n")
                
                if 'optimal_threshold' in data:
                    f.write(f"   Optimal Threshold: {data['optimal_threshold']:.3f}\n")
                    f.write(f"   Optimal F1:        {data['optimal_f1']:.3f}\n")



def main():
    # Define comparisons
    comparator = Comparator()

    