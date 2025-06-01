# src/system/modules/EvaluationModule.py

from ..SystemModule import SystemModule
from ..SystemData import SystemData
from ..structs.Detection import Detection
from config import Config

import logging
from datetime import datetime
import os
from collections import defaultdict
from typing import Set, Dict, Any, Optional, List, Tuple
import numpy as np
import matplotlib.pyplot as plt
import json
import seaborn as sns


class EvaluationModule(SystemModule):
    """
    Collects and evaluates detection results
    Saves results, visualizations, and annotated images
    """
    def __init__(self, 
                 config: Config, 
                 module_name: str = "EvaluationModule"):
        """Initialize module"""
        super().__init__(config, module_name)
        self.logger = logging.getLogger(self.name)

        timestamp = datetime.now().strftime("%m%d_%H%M%S")
        self.output_dir = os.path.join(self._config.eval_output_folder, timestamp)

        self.results = defaultdict(lambda: defaultdict(list))
        self.ground_truth_data = []
        self.image_paths = []
        self._process_data = None
        self.treshold_metrics = defaultdict(lambda: defaultdict(dict))

        self.iou_threshold = 0.5
        self.confidence_thresholds = [0.1, 0.3, 0.5, 0.7, 0.9]
        self.eval_thresholds = np.arange(0.0, 1.0, 0.05)

        self.detection_sources = self._config.eval_detection_sources
        self.depth_sources = [
            SystemData.DEPTH,
            SystemData.MIDAS_ESTIMATED_DEPTH,
            SystemData.PRO_ESTIMATED_DEPTH,
            SystemData.VGGT_ESTIMATED_DEPTH,
            SystemData.NORM_DEPTH,
            SystemData.NORM_MIDAS,
            SystemData.NORM_PRO,
            SystemData.NORM_VGGT
        ]

        self.is_initialized = False
        self.frame_count = 0


    def initialize(self, config) -> bool:
        """Initialize module"""
        if self.is_initialized:
            self.logger.info(f"{self.name} already initialized")
            return True
        
        try:
            os.makedirs(self.output_dir, exist_ok=True)
            os.makedirs(os.path.join(self.output_dir, "annotated_images"), exist_ok=True)
            os.makedirs(os.path.join(self.output_dir, "depth_analysis"), exist_ok=True)
            os.makedirs(os.path.join(self.output_dir, "plots"), exist_ok=True)
        
            self.logger.info(f"Evaluation output directory: {self.output_dir}")
            self.is_initialized = True
            return True
            
        except Exception as e:
            self.logger.error(f"Error initializing {self.name}: {e}")
            return False
        
    def get_required_inputs(self) -> Set[str]:
        return set()

    def get_dependency_inputs(self) -> Set[str]:
        dependencies = {
            SystemData.COLOR,
            SystemData.GROUND_TRUTH_DETECTIONS,
            SystemData.GROUND_TRUTH_IMAGE_PATH,
            SystemData.EVAL_COMPLETE
        }
        dependencies.update(self.detection_sources)
        dependencies.update(self.depth_sources)
        return dependencies
    
    def get_outputs(self) -> Set[str]:
        return {SystemData.EVAL_RESULTS, SystemData.EVAL_METRICS}
    
    def _process_internal(self, data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Collect detecion results for each frame"""
        if not self.is_initialized:
            return None
    
        if data.get(SystemData.EVAL_COMPLETE):
            self.logger.info("Evaluation complete signal received")
            self._finalize_evaluation()
            return {SystemData.EVAL_METRICS: "Evaluation completed"}
        
        ground_truth = data.get(SystemData.GROUND_TRUTH_DETECTIONS, [])
        image_path = data.get(SystemData.GROUND_TRUTH_IMAGE_PATH, f"frame_{self.frame_count}")
        color_image = data.get(SystemData.COLOR)

        self.ground_truth_data.append(ground_truth)
        self.image_paths.append(image_path)

        frame_results = {}
        for source in self.detection_sources:
            if source in data:
                detections = data[source]
                frame_results[source] = detections
                self.results[source]['detections'].append(detections)

        if color_image is not None:
            self._process_data = data
            self._save_annotated_image(color_image, ground_truth, frame_results, image_path, data)
        
        self._analyze_depth_maps(data, image_path)

        self.frame_count += 1

        return {SystemData.EVAL_RESULTS: frame_results}
    
    def _analyze_depth_maps(self, data: Dict[str, Any], image_name: str):
        """Rudamentary depth map analysis"""
        depth_maps = {}
        for source in self.depth_sources:
            if source in data and data[source] is not None:
                depth_maps[source] = data[source]

        if len(depth_maps) < 2:
            return
        
        stats = {}
        for name, depth_map in depth_maps.items():
            if depth_map is not None and depth_map.size > 0:
                valid_mask = depth_map > 0
                if np.any(valid_mask):
                    valid_depths = depth_map[valid_mask]
                    stats[name] = {
                        'mean': np.mean(valid_depths),
                        'std': np.std(valid_depths),
                        'valid_ratio': np.sum(valid_mask) / depth_map.size
                    }

        self.results['depth_stats'][image_name] = stats

        if len(depth_maps) > 1:
            self._save_depth_comparison(depth_maps, image_name)

    def _save_depth_comparison(self, depth_maps: Dict[str, np.ndarray], image_name: str):
        """Save depth map comparison visualization"""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        axes = axes.ravel()
        
        for idx, (name, depth_map) in enumerate(depth_maps.items()):
            if idx < len(axes):
                ax = axes[idx]
                
                # Normalize for visualization
                valid_mask = depth_map > 0
                if np.any(valid_mask):
                    vmin = np.percentile(depth_map[valid_mask], 5)
                    vmax = np.percentile(depth_map[valid_mask], 95)
                    
                    im = ax.imshow(depth_map, cmap='viridis', vmin=vmin, vmax=vmax)
                    ax.set_title(name.replace('_', ' ').title())
                    ax.axis('off')
                    plt.colorbar(im, ax=ax, fraction=0.046)
        
        # Hide unused subplots
        for idx in range(len(depth_maps), len(axes)):
            axes[idx].axis('off')
        
        plt.tight_layout()
        output_path = os.path.join(self.output_dir, "depth_analysis", f"{image_name}_depth_comparison.png")
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close() 

    def _finalize_evaluation(self):
        """Compute final metrics and save result"""
        self.logger.info(f"Computing evaluation metrics")

        metrics = {}

        # Compute metrics for each detection source
        for source in self.detection_sources:
            if source in self.results and self.results[source]['detections']:
                source_metrics = self._compute_detection_metrics(
                    self.results[source]['detections'],
                    self.ground_truth_data,
                )
                metrics[source] = source_metrics

        # Compute analyzis on depth maps
        depth_metrics = self._compute_depth_metrics()
        metrics['depth_metrics'] = depth_metrics

        # Save results
        self._save_metrics(metrics)
        self._generate_plots(metrics)

        self.logger.info(f"Evaluation complete, results saved to {self.output_dir}")

    def _compute_detection_metrics(self, predictions: List[List[Detection]], ground_truths: List[List[Detection]]) -> Dict[str, Any]:
        """Compute object detection metrics"""
        all_predictions = []
        all_ground_truths = []

        for frame_preds, frame_gt in zip(predictions, ground_truths):
            all_predictions.extend(frame_preds)
            all_ground_truths.extend(frame_gt)


        metrics = {
            'total_predictions': len(all_predictions),
            'total_ground_truth': len(all_ground_truths),
            'per_class_metrics': {},
            'overall_metrics': {}
        }
        
        # Compute metrics for different classes
        all_classes = set()
        for det in all_predictions + all_ground_truths:
            all_classes.add(det.class_id)
        
        for class_id in all_classes:
            class_preds = [d for d in all_predictions if d.class_id == class_id]
            class_gt = [d for d in all_ground_truths if d.class_id == class_id]

            if class_gt:
                class_metrics = self._compute_class_metrics(class_preds, class_gt)
                class_name = self._config.class_names.get(class_id, f"class_{class_id}")
                metrics['per_class_metrics'][class_name] = class_metrics

        # Compute global mettrics
        if all_ground_truths:
            overall_metrics = self._compute_class_metrics(all_predictions, all_ground_truths)
            metrics['overall_metrics'] = overall_metrics
        
         # Add threshold analysis
        all_predictions_flat = []
        all_ground_truths_flat = []
        for frame_preds, frame_gt in zip(predictions, ground_truths):
            all_predictions_flat.extend(frame_preds)
            all_ground_truths_flat.extend(frame_gt)
        
        # Compute metrics at multiple thresholds
        threshold_metrics = self._compute_metrics_at_thresholds(all_predictions_flat, all_ground_truths_flat)
        metrics['threshold_analysis'] = threshold_metrics
        
        # Find optimal threshold (highest F1 score)
        best_threshold = 0.5
        best_f1 = 0
        for thresh_data in threshold_metrics.values():
            if thresh_data['f1_score'] > best_f1:
                best_f1 = thresh_data['f1_score']
                best_threshold = thresh_data['threshold']
        
        metrics['optimal_threshold'] = best_threshold
        metrics['optimal_f1'] = best_f1
        
        return metrics
    

    def _compute_class_metrics(self, predictions: List[Detection], ground_truths: List[Detection]) -> Dict[str, Any]:
        """Compute metrics for a single class"""
        metrics = {
            'precision': 0.0,
            'recall': 0.0,
            'f1_score': 0.0,
            'average_precision': 0.0,
            'precision_recall_curve': {'precisions': [], 'recalls': []},
            'confusion_matrix': {'tp': 0, 'fp': 0, 'fn': 0, 'tn': 0}
        }

        if not ground_truths:
            # If no ground truth, all predictions are false positives
            metrics['confusion_matrix']['fp'] = len(predictions)
            return metrics

        # Sort predictions by confidence
        sorted_preds = sorted(predictions, key=lambda x: x.conf, reverse=True)

        # Store all PR points
        all_precisions = []
        all_recalls = []
        
        # For confusion matrix at operating threshold
        tp_at_threshold = 0
        fp_at_threshold = 0
        fn_at_threshold = 0

        # Compute metrics at different confidence thresholds
        for conf_threshold in np.linspace(0, 1, 101):
            # Filter predictions by confidence
            filtered_preds = [p for p in sorted_preds if p.conf >= conf_threshold]
            
            # Match predictions to ground truth
            tp, fp, fn = self._match_detections(filtered_preds, ground_truths)
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 1.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            
            all_precisions.append(precision)
            all_recalls.append(recall)
            
            # Store values at operating threshold for confusion matrix
            if abs(conf_threshold - self._config.confidence_threshold) < 0.01:
                tp_at_threshold = tp
                fp_at_threshold = fp
                fn_at_threshold = fn
        
        # Store full PR curve
        metrics['precision_recall_curve']['precisions'] = all_precisions
        metrics['precision_recall_curve']['recalls'] = all_recalls
        
        # Compute average precision using all points
        if all_precisions and all_recalls:
            # Sort by recall
            sorted_indices = np.argsort(all_recalls)
            sorted_recalls = [all_recalls[i] for i in sorted_indices]
            sorted_precisions = [all_precisions[i] for i in sorted_indices]
            
            # Compute AP using trapezoidal rule
            ap = 0
            for i in range(1, len(sorted_recalls)):
                ap += (sorted_recalls[i] - sorted_recalls[i-1]) * (sorted_precisions[i] + sorted_precisions[i-1]) / 2
            
            metrics['average_precision'] = ap
            
            # Get metrics at specified confidence threshold
            idx = int(self._config.confidence_threshold * 100)
            if idx < len(all_precisions):
                metrics['precision'] = all_precisions[idx]
                metrics['recall'] = all_recalls[idx]
                if all_precisions[idx] + all_recalls[idx] > 0:
                    metrics['f1_score'] = 2 * (all_precisions[idx] * all_recalls[idx]) / (all_precisions[idx] + all_recalls[idx])
        
        # Store confusion matrix values
        metrics['confusion_matrix']['tp'] = tp_at_threshold
        metrics['confusion_matrix']['fp'] = fp_at_threshold
        metrics['confusion_matrix']['fn'] = fn_at_threshold
        # TN is not meaningful for object detection
        
        return metrics
    
    def _match_detections(self, predictions: List[Detection], ground_truths: List[Detection]) -> Tuple[int, int, int]:
        """Match detections to ground truth"""
        tp = 0
        fp = 0
        fn = 0

        matched_gt = set()

        for pred in predictions:
            best_iou = 0
            best_gt_idx = -1

            for gt_idx, gt in enumerate(ground_truths):
                if gt_idx  in matched_gt:
                    continue

                iou = self._compute_iou(pred.bbox2D, gt.bbox2D)
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = gt_idx

            if best_iou >= self.iou_threshold:
                tp += 1
                matched_gt.add(best_gt_idx)
            else:
                fp += 1
        
        fn = len(ground_truths) - len(matched_gt)

        return tp, fp, fn

    def _compute_iou(self, box1: List[float], box2: List[float]) -> float:
        """Compute IoU between two bounding boxes"""
        if not box1 or not box2:
            return 0.0
        
        x1_1, y1_1, x2_1, y2_1 = box1
        x1_2, y1_2, x2_2, y2_2 = box2

        # Find intersection points
        x1_i = max(x1_1, x1_2)
        y1_i = max(y1_1, y1_2)
        x2_i = min(x2_1, x2_2)
        y2_i = min(y2_1, y2_2)

        if x2_i < x1_i or y2_i < y1_i:
            return 0.0
        
        intersection = (x2_i - x1_i) * (y2_i - y1_i)

        # Compute union
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        union = area1 + area2 - intersection

        return intersection/union if union > 0 else 0.0
    
    def _compute_metrics_at_thresholds(self, predictions: List[Detection], ground_truths: List[Detection]) -> Dict[str, Any]:
        """Compute metrics at multiple confidence thresholds"""
        threshold_results = {}
        
        for threshold in self.eval_thresholds:
            # Filter predictions by threshold
            filtered_preds = [p for p in predictions if p.conf >= threshold]
            
            # Compute metrics
            tp, fp, fn = self._match_detections(filtered_preds, ground_truths)
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 1.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
            
            threshold_results[f'thresh_{threshold:.2f}'] = {
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'tp': tp,
                'fp': fp,
                'fn': fn,
                'threshold': threshold
            }
        
        return threshold_results

    def _compute_depth_metrics(self) -> Dict[str, Any]:
        """Analyze depth maps"""
        metrics = {}

        if 'depth_stats' not in self.results:
            return metrics
        
        # Aggregate statistics across all frames
        aggregated_stats = defaultdict(lambda: defaultdict(list))

        for frame_stats in self.results['depth_stats'].values():
            for source, stats in frame_stats.items():
                for metric, value in stats.items():
                    aggregated_stats[source][metric].append(value)

        # Compute summary statistics
        for source, stats in aggregated_stats.items():
            source_metrics = {}
            for metric, values in stats.items():
                if values:  # Check if list is not empty
                    source_metrics[f"{metric}_mean"] = np.mean(values)
                    source_metrics[f"{metric}_std"] = np.std(values)    

            metrics[source] = source_metrics

        return metrics
    
    def _save_metrics(self, metrics: Dict[str, Any]):
        """Save metrics to JSON file"""
        output_path = os.path.join(self.output_dir, "evaluation_metrics.json")
        
        # Convert numpy types to Python types for JSON serialization
        def convert_types(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, (np.float32, np.float64)):
                return float(obj)
            elif isinstance(obj, (np.int32, np.int64)):
                return int(obj)
            elif isinstance(obj, dict):
                return {k: convert_types(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_types(v) for v in obj]
            return obj
        
        metrics_serializable = convert_types(metrics)
        
        with open(output_path, 'w') as f:
            json.dump(metrics_serializable, f, indent=2)
        
        # Also save a summary text file
        summary_path = os.path.join(self.output_dir, "evaluation_summary.txt")
        with open(summary_path, 'w') as f:
            f.write("EVALUATION SUMMARY\n")
            f.write("=" * 50 + "\n\n")
            
            # Detection metrics summary
            f.write("OBJECT DETECTION METRICS\n")
            f.write("-" * 30 + "\n")
            
            for source, source_metrics in metrics.items():
                if source != 'depth_metrics' and isinstance(source_metrics, dict):
                    f.write(f"\n{source}:\n")
                    if 'overall_metrics' in source_metrics:
                        overall = source_metrics['overall_metrics']
                        # Check if values are numbers before formatting
                        ap = overall.get('average_precision', 0)
                        prec = overall.get('precision', 0)
                        rec = overall.get('recall', 0)
                        f1 = overall.get('f1_score', 0)
                        
                        f.write(f"  Average Precision: {float(ap):.3f}\n")
                        f.write(f"  Precision: {float(prec):.3f}\n")
                        f.write(f"  Recall: {float(rec):.3f}\n")
                        f.write(f"  F1 Score: {float(f1):.3f}\n")
            
            # Depth analysis summary
            if 'depth_metrics' in metrics:
                f.write("\n\nDEPTH ESTIMATION ANALYSIS\n")
                f.write("-" * 30 + "\n")
                
                depth_metrics = metrics['depth_metrics']
                for source, source_metrics in depth_metrics.items():
                    if isinstance(source_metrics, dict):
                        f.write(f"\n{source}:\n")
                        for metric, value in source_metrics.items():
                            f.write(f"  {metric}: {float(value):.3f}\n")
    
    def _generate_plots(self, metrics: Dict[str, Any]):
        """Generate all evaluation plots"""
        os.makedirs(os.path.join(self.output_dir, "plots"), exist_ok=True)
        
        try:
            self._plot_ap_comparison(metrics)
        except Exception as e:
            self.logger.error(f"Error creating AP comparison plot: {e}")
        
        try:
            self._plot_pr_curves(metrics)
        except Exception as e:
            self.logger.error(f"Error creating PR curves: {e}")
        
        try:
            self._plot_depth_consistency(metrics)
        except Exception as e:
            self.logger.error(f"Error creating depth consistency plot: {e}")
        
        try:
            self._plot_per_class_performance(metrics)
        except Exception as e:
            self.logger.error(f"Error creating per-class performance plot: {e}")
        
        try:
            self._plot_confusion_matrix(metrics)
        except Exception as e:
            self.logger.error(f"Error creating confusion matrices: {e}")
        
        try:
            self._plot_metrics_summary(metrics)
        except Exception as e:
            self.logger.error(f"Error creating metrics summary: {e}")
        
        try:
            self._plot_threshold_analysis(metrics)
        except Exception as e:
            self.logger.error(f"Error creating threshold analysis plot: {e}")
    

    def _plot_metrics_summary(self, metrics: Dict[str, Any]):
        """Plot a summary of key metrics across all sources"""
        sources = []
        precisions = []
        recalls = []
        f1_scores = []
        
        for source, source_metrics in metrics.items():
            if source != 'depth_metrics' and 'overall_metrics' in source_metrics:
                overall = source_metrics['overall_metrics']
                sources.append(source.replace('_', ' ').title())
                precisions.append(overall.get('precision', 0))
                recalls.append(overall.get('recall', 0))
                f1_scores.append(overall.get('f1_score', 0))
        
        if not sources:
            return
        
        x = np.arange(len(sources))
        width = 0.25
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        bars1 = ax.bar(x - width, precisions, width, label='Precision', color='skyblue')
        bars2 = ax.bar(x, recalls, width, label='Recall', color='lightcoral')
        bars3 = ax.bar(x + width, f1_scores, width, label='F1 Score', color='lightgreen')
        
        ax.set_xlabel('Detection Source')
        ax.set_ylabel('Score')
        ax.set_title('Detection Performance Metrics Summary')
        ax.set_xticks(x)
        ax.set_xticklabels(sources, rotation=45, ha='right')
        ax.legend()
        ax.set_ylim(0, 1.1)
        
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
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "plots", "metrics_summary.png"), dpi=150)
        plt.close()

    def _plot_ap_comparison(self, metrics: Dict[str, Any]):
        """Plot Average Precision comparison across sources"""
        sources = []
        aps = []
        
        for source, source_metrics in metrics.items():
            if source != 'depth_analysis' and 'overall_metrics' in source_metrics:
                sources.append(source.replace('_', ' ').title())
                aps.append(source_metrics['overall_metrics'].get('average_precision', 0))
        
        if not sources:
            return
        
        plt.figure(figsize=(10, 6))
        bars = plt.bar(sources, aps)
        
        # Color bars based on value
        for bar, ap in zip(bars, aps):
            if ap > 0.7:
                bar.set_color('green')
            elif ap > 0.5:
                bar.set_color('orange')
            else:
                bar.set_color('red')
        
        plt.ylabel('Average Precision')
        plt.title('Object Detection Performance Comparison')
        plt.xticks(rotation=45, ha='right')
        plt.ylim(0, 1)
        
        # Add value labels on bars
        for i, (bar, ap) in enumerate(zip(bars, aps)):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{ap:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "plots", "ap_comparison.png"), dpi=150)
        plt.close()
    
    def _plot_threshold_analysis(self, metrics: Dict[str, Any]):
        """Plot precision, recall, and F1 vs confidence threshold"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        for idx, (source, source_metrics) in enumerate(metrics.items()):
            if source != 'depth_metrics' and 'threshold_analysis' in source_metrics:
                ax = axes[idx // 2, idx % 2]
                
                thresh_data = source_metrics['threshold_analysis']
                thresholds = []
                precisions = []
                recalls = []
                f1_scores = []
                
                for thresh_key in sorted(thresh_data.keys()):
                    data = thresh_data[thresh_key]
                    thresholds.append(data['threshold'])
                    precisions.append(data['precision'])
                    recalls.append(data['recall'])
                    f1_scores.append(data['f1_score'])
                
                ax.plot(thresholds, precisions, 'b-', label='Precision', linewidth=2)
                ax.plot(thresholds, recalls, 'r-', label='Recall', linewidth=2)
                ax.plot(thresholds, f1_scores, 'g-', label='F1 Score', linewidth=2)
                
                # Mark optimal threshold
                optimal_thresh = source_metrics.get('optimal_threshold', 0.5)
                ax.axvline(x=optimal_thresh, color='black', linestyle='--', alpha=0.7, label=f'Optimal ({optimal_thresh:.2f})')
                
                # Mark current threshold
                ax.axvline(x=0.75, color='red', linestyle=':', alpha=0.7, label='Current (0.75)')
                
                ax.set_xlabel('Confidence Threshold')
                ax.set_ylabel('Score')
                ax.set_title(f'{source.replace("_", " ").title()}')
                ax.legend()
                ax.grid(True, alpha=0.3)
                ax.set_xlim(0, 1)
                ax.set_ylim(0, 1.1)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "plots", "threshold_analysis.png"), dpi=150)
        plt.close()

    def _plot_pr_curves(self, metrics: Dict[str, Any]):
        """Plot Precision-Recall curves for each detection source"""
        plt.figure(figsize=(12, 8))
        
        colors = ['blue', 'red', 'green', 'orange', 'purple']
        line_styles = ['-', '--', '-.', ':']
        
        plot_idx = 0
        for source, source_metrics in metrics.items():
            if source != 'depth_metrics' and 'overall_metrics' in source_metrics:
                overall = source_metrics['overall_metrics']
                if 'precision_recall_curve' in overall:
                    pr_data = overall['precision_recall_curve']
                    precisions = pr_data.get('precisions', [])
                    recalls = pr_data.get('recalls', [])
                    
                    if len(precisions) > 1 and len(recalls) > 1:
                        # Sort by recall for proper curve
                        sorted_pairs = sorted(zip(recalls, precisions))
                        sorted_recalls, sorted_precisions = zip(*sorted_pairs)
                        
                        source_name = source.replace('_detections', '').replace('_', ' ').title()
                        ap = overall.get('average_precision', 0)
                        
                        plt.plot(sorted_recalls, sorted_precisions, 
                                color=colors[plot_idx % len(colors)],
                                linestyle=line_styles[plot_idx % len(line_styles)],
                                label=f'{source_name} (AP={ap:.3f})',
                                linewidth=2)
                        plot_idx += 1
        
        plt.xlabel('Recall', fontsize=12)
        plt.ylabel('Precision', fontsize=12)
        plt.title('Precision-Recall Curves', fontsize=14)
        plt.legend(loc='lower left', fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.xlim([0, 1])
        plt.ylim([0, 1.05])
        
        # Add diagonal reference line
        plt.plot([0, 1], [0.5, 0.5], 'k--', alpha=0.3, label='Random classifier')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "plots", "precision_recall_curves.png"), dpi=150, bbox_inches='tight')
        plt.close()
    
    def _plot_depth_consistency(self, metrics: Dict[str, Any]):
        """Plot depth estimation consistency heatmap"""
        if 'depth_metrics' not in metrics:
            return
        
        depth_sources = ['depth', 'midas_estimated_depth', 'pro_estimated_depth', 'vggt_estimated_depth']
        display_names = ['Camera', 'MiDaS', 'Depth Pro', 'VGGT']
        
        # Create correlation matrix based on valid pixel ratios
        n_sources = len(depth_sources)
        consistency_matrix = np.ones((n_sources, n_sources))
        
        depth_metrics = metrics['depth_metrics']
        
        # Fill matrix with normalized differences
        for i, source1 in enumerate(depth_sources):
            for j, source2 in enumerate(depth_sources):
                if i != j and source1 in depth_metrics and source2 in depth_metrics:
                    # Use valid_ratio as a proxy for consistency
                    valid1 = depth_metrics[source1].get('valid_ratio_mean', 0)
                    valid2 = depth_metrics[source2].get('valid_ratio_mean', 0)
                    
                    # Calculate similarity based on valid pixel ratios
                    consistency_matrix[i, j] = 1 - abs(valid1 - valid2)
        
        # Plot heatmap
        plt.figure(figsize=(8, 6))
        sns.heatmap(consistency_matrix, 
                xticklabels=display_names,
                yticklabels=display_names,
                annot=True, fmt='.3f', cmap='RdYlGn', 
                vmin=0, vmax=1,
                cbar_kws={'label': 'Consistency Score'})
        
        plt.title('Depth Estimation Consistency')
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "plots", "depth_consistency.png"), dpi=150)
        plt.close()

    def _plot_confusion_matrix(self, metrics: Dict[str, Any]):
        """Plot confusion matrices for each detection source"""
        # First collect actual confusion data
        all_sources_data = []
        source_names = []
        
        for source, source_metrics in metrics.items():
            if source != 'depth_metrics' and 'per_class_metrics' in source_metrics:
                source_names.append(source.replace('_detections', '').replace('_', ' ').title())
                
                # Build confusion matrix from per-class metrics
                class_names = list(self._config.class_names.values())
                n_classes = len(class_names)
                
                # Create matrix: rows = actual, cols = predicted
                conf_matrix = np.zeros((n_classes, n_classes))
                
                for i, class_name in enumerate(class_names):
                    if class_name in source_metrics['per_class_metrics']:
                        class_data = source_metrics['per_class_metrics'][class_name]
                        cm = class_data.get('confusion_matrix', {})
                        
                        # Get TP for this class
                        tp = cm.get('tp', 0)
                        fp = cm.get('fp', 0)
                        fn = cm.get('fn', 0)
                        
                        # Place in matrix
                        conf_matrix[i, i] = tp  # True positives on diagonal
                        
                        # Distribute FP and FN (this is approximate without full data)
                        if n_classes > 1:
                            # Distribute false positives to other classes
                            for j in range(n_classes):
                                if i != j:
                                    conf_matrix[j, i] += fp / (n_classes - 1)
                            
                            # Distribute false negatives
                            for j in range(n_classes):
                                if i != j:
                                    conf_matrix[i, j] += fn / (n_classes - 1)
                
                all_sources_data.append(conf_matrix)
        
        if not all_sources_data:
            return
        
        # Create subplots
        n_sources = len(all_sources_data)
        cols = min(2, n_sources)
        rows = (n_sources + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(6*cols, 5*rows))
        if n_sources == 1:
            axes = [axes]
        else:
            axes = axes.ravel()
        
        for idx, (conf_matrix, source_name) in enumerate(zip(all_sources_data, source_names)):
            if idx < len(axes):
                ax = axes[idx]
                
                # Normalize to percentages
                row_sums = conf_matrix.sum(axis=1, keepdims=True)
                row_sums[row_sums == 0] = 1  # Avoid division by zero
                conf_matrix_norm = conf_matrix / row_sums * 100
                
                # Plot
                im = ax.imshow(conf_matrix_norm, cmap='Blues', vmin=0, vmax=100)
                
                # Set ticks
                class_names = list(self._config.class_names.values())
                ax.set_xticks(range(len(class_names)))
                ax.set_yticks(range(len(class_names)))
                ax.set_xticklabels(class_names, rotation=45, ha='right')
                ax.set_yticklabels(class_names)
                
                # Labels
                ax.set_xlabel('Predicted', fontsize=10)
                ax.set_ylabel('Actual', fontsize=10)
                ax.set_title(f'{source_name}', fontsize=12)
                
                # Add text annotations
                for i in range(len(class_names)):
                    for j in range(len(class_names)):
                        value = conf_matrix_norm[i, j]
                        text_color = 'white' if value > 50 else 'black'
                        text = ax.text(j, i, f'{value:.1f}%',
                                    ha='center', va='center',
                                    color=text_color, fontsize=9)
                
                # Add colorbar
                plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        
        # Hide unused subplots
        for idx in range(n_sources, len(axes)):
            axes[idx].axis('off')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "plots", "confusion_matrices.png"), dpi=150, bbox_inches='tight')
        plt.close()
    
    def _plot_per_class_performance(self, metrics: Dict[str, Any]):
        """Plot per-class performance comparison"""
        # Collect per-class APs for each source
        class_names = list(self.config.class_names.values())
        sources = []
        class_aps = defaultdict(list)
        
        for source, source_metrics in metrics.items():
            if source != 'depth_analysis' and 'per_class_metrics' in source_metrics:
                sources.append(source.replace('_', ' ').title())
                
                for class_name in class_names:
                    if class_name in source_metrics['per_class_metrics']:
                        ap = source_metrics['per_class_metrics'][class_name].get('average_precision', 0)
                    else:
                        ap = 0
                    class_aps[class_name].append(ap)
        
        if not sources or not class_aps:
            return
        
        # Create grouped bar chart
        x = np.arange(len(sources))
        width = 0.8 / len(class_names)
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        for i, class_name in enumerate(class_names):
            offset = width * (i - len(class_names)/2 + 0.5)
            ax.bar(x + offset, class_aps[class_name], width, label=class_name)
        
        ax.set_xlabel('Detection Source')
        ax.set_ylabel('Average Precision')
        ax.set_title('Per-Class Detection Performance')
        ax.set_xticks(x)
        ax.set_xticklabels(sources, rotation=45, ha='right')
        ax.legend()
        ax.set_ylim(0, 1)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "plots", "per_class_performance.png"), dpi=150)
        plt.close()
    
    
    def _save_annotated_image(self, image: np.ndarray, ground_truth: List[Detection], 
                        detections: Dict[str, List[Detection]], image_name: str,
                        current_data: Dict[str, Any]):
        """Save image with ground truth and detection overlays"""
        import cv2
        
        # Create subdirectories for different overlay types
        subdirs = [
            'color_with_gt',
            'color_with_detections', 
            'normalized_depths',
            'depth_with_detections',
            'depth_with_gt',
            'montages',
            'comparisons'
        ]
        
        for subdir in subdirs:
            os.makedirs(os.path.join(self.output_dir, "annotated_images", subdir), exist_ok=True)
        
        # Get normalized depth images from current data
        normalized_depths = {}
        
        # Map normalized depth keys to their sources
        norm_mapping = {
            SystemData.NORM_DEPTH: 'depth',
            SystemData.NORM_MIDAS: 'midas', 
            SystemData.NORM_PRO: 'pro',
            SystemData.NORM_VGGT: 'vggt'
        }
        
        self.logger.debug(f"Available data keys: {current_data.keys()}")
        
        for norm_key, name in norm_mapping.items():
            if norm_key in current_data and current_data[norm_key] is not None:
                normalized_depths[name] = current_data[norm_key]
                self.logger.debug(f"Found normalized depth for {name}")
        
        self.logger.info(f"Found {len(normalized_depths)} normalized depth images for {image_name}")
        
        base_name = os.path.splitext(image_name)[0]
        annotated_images = []
        
        # Helper function to add text with background in corner
        def put_text_with_background(img, text, font_scale=0.7, thickness=2):
            font = cv2.FONT_HERSHEY_SIMPLEX
            text_size, _ = cv2.getTextSize(text, font, font_scale, thickness)
            text_w, text_h = text_size
            
            # Position at exact corner
            x = 0
            y = text_h +5  # Small offset from top
            
            # Draw black background rectangle from corner
            cv2.rectangle(img, (0, 0), (text_w + 10, y + 5), (0, 0, 0), -1)
            
            # Draw white text
            cv2.putText(img, text, (5, y), font, font_scale, (255, 255, 255), thickness)
            return img
        
        # 1. Save color image with ground truth
        gt_color_image = image.copy()
        gt_color_image = self._draw_detections_with_colors(gt_color_image, ground_truth, 
                                                        is_ground_truth=True)
        gt_color_image = put_text_with_background(gt_color_image, "Ground Truth")
        cv2.imwrite(os.path.join(self.output_dir, "annotated_images", "color_with_gt", f"{base_name}.png"), 
                    gt_color_image)
        annotated_images.append(("Ground Truth (Color)", gt_color_image))
        
        # 2. Save raw normalized depth images
        for depth_name, norm_depth in normalized_depths.items():
            # Convert to viewable format
            if norm_depth.dtype == np.uint16:
                # Scale to 8-bit for visualization
                norm_depth_vis = (norm_depth / 256).astype(np.uint8)
            else:
                norm_depth_vis = norm_depth.astype(np.uint8) if norm_depth.dtype != np.uint8 else norm_depth
                
            output_path = os.path.join(self.output_dir, "annotated_images", "normalized_depths", 
                                    f"{base_name}_{depth_name}_normalized.png")
            cv2.imwrite(output_path, norm_depth_vis)
            self.logger.debug(f"Saved normalized depth to {output_path}")
        
        # 3. Save each detection source on its corresponding normalized depth
        detection_source_mapping = {
            SystemData.DEPTH_DETECTIONS: ('depth', 'Camera Depth'),
            SystemData.MIDAS_DETECTIONS: ('midas', 'MiDaS'),
            SystemData.PRO_DETECTIONS: ('pro', 'Depth Pro'), 
            SystemData.VGGT_DETECTIONS: ('vggt', 'VGGT')
        }
        
        for detection_source, (depth_key, display_name) in detection_source_mapping.items():
            if detection_source in detections and depth_key in normalized_depths:
                # Get the normalized depth image
                norm_depth = normalized_depths[depth_key]
                
                # Convert to 3-channel for colored annotations
                if norm_depth.dtype == np.uint16:
                    norm_depth_8bit = (norm_depth / 256).astype(np.uint8)
                else:
                    norm_depth_8bit = norm_depth.astype(np.uint8) if norm_depth.dtype != np.uint8 else norm_depth
                    
                if len(norm_depth_8bit.shape) == 2:
                    norm_depth_vis = cv2.cvtColor(norm_depth_8bit, cv2.COLOR_GRAY2BGR)
                else:
                    norm_depth_vis = norm_depth_8bit.copy()
                
                # Draw detections with class colors
                dets = detections[detection_source]
                norm_depth_vis = self._draw_detections_with_colors(norm_depth_vis, dets, is_ground_truth=False)
                norm_depth_vis = put_text_with_background(norm_depth_vis, f"{display_name} Detections")
                
                # Save to appropriate folder
                output_path = os.path.join(self.output_dir, "annotated_images", "depth_with_detections",
                                        f"{base_name}_{depth_key}.png")
                cv2.imwrite(output_path, norm_depth_vis)
                self.logger.debug(f"Saved depth with detections to {output_path}")
                
                annotated_images.append((f"{display_name} Detections", norm_depth_vis))
                
                # Also save normalized depth with ground truth
                norm_depth_gt_vis = cv2.cvtColor(norm_depth_8bit, cv2.COLOR_GRAY2BGR) if len(norm_depth_8bit.shape) == 2 else norm_depth_8bit.copy()
                norm_depth_gt_vis = self._draw_detections_with_colors(norm_depth_gt_vis, ground_truth, 
                                                                    is_ground_truth=True)
                norm_depth_gt_vis = put_text_with_background(norm_depth_gt_vis, f"{display_name} w/ GT")
                
                output_path_gt = os.path.join(self.output_dir, "annotated_images", "depth_with_gt",
                                            f"{base_name}_{depth_key}.png")
                cv2.imwrite(output_path_gt, norm_depth_gt_vis)
                self.logger.debug(f"Saved depth with GT to {output_path_gt}")
        
        # 4. Save detection overlays on color image
        for source, dets in detections.items():
            if dets:  # Only save if there are detections
                color_det_image = image.copy()
                color_det_image = self._draw_detections_with_colors(color_det_image, dets, is_ground_truth=False)
                source_name = source.replace("_detections", "").replace("_", " ").title()
                color_det_image = put_text_with_background(color_det_image, source_name)
                
                cv2.imwrite(os.path.join(self.output_dir, "annotated_images", "color_with_detections",
                                    f"{base_name}_{source}.png"), color_det_image)
                
                annotated_images.append((source_name, color_det_image))
        
        # 5. Create comparison montages
        if len(annotated_images) >= 2:
            # Create 2x2 comparison of normalized depths with detections
            comparison_images = []
            
            for detection_source, (depth_key, display_name) in detection_source_mapping.items():
                if detection_source in detections and depth_key in normalized_depths:
                    # Find the corresponding annotated image
                    for title, img in annotated_images:
                        if display_name in title and "Detections" in title:
                            comparison_images.append(img)
                            break
            
            if len(comparison_images) >= 2:
                h, w = comparison_images[0].shape[:2]
                rows = 2
                cols = 2
                comp_canvas = np.zeros((rows * h, cols * w, 3), dtype=np.uint8)
                
                for idx, img in enumerate(comparison_images[:4]):
                    row = idx // cols
                    col = idx % cols
                    y1 = row * h
                    y2 = (row + 1) * h
                    x1 = col * w
                    x2 = (col + 1) * w
                    comp_canvas[y1:y2, x1:x2] = img
                
                cv2.imwrite(os.path.join(self.output_dir, "annotated_images", "comparisons",
                                    f"{base_name}_comparison.png"), comp_canvas)
            
            # Create full montage
            if annotated_images:
                n_images = len(annotated_images)
                cols = min(3, n_images)
                rows = (n_images + cols - 1) // cols
                
                h, w = annotated_images[0][1].shape[:2]
                canvas = np.zeros((rows * h, cols * w, 3), dtype=np.uint8)
                
                for idx, (title, img) in enumerate(annotated_images):
                    row = idx // cols
                    col = idx % cols
                    y1 = row * h
                    y2 = (row + 1) * h
                    x1 = col * w
                    x2 = (col + 1) * w
                    canvas[y1:y2, x1:x2] = img
                
                cv2.imwrite(os.path.join(self.output_dir, "annotated_images", "montages",
                                    f"{base_name}_all.png"), canvas)

    def _draw_detections_with_colors(self, image: np.ndarray, detections: List[Detection], 
                                    is_ground_truth: bool = False) -> np.ndarray:
        """Draw detections with class-specific colors from config"""
        import cv2
        
        output_image = image.copy()
        
        for det in detections:
            if det.bbox2D:
                x1, y1, x2, y2 = [int(v) for v in det.bbox2D]
                
                # Get color based on class
                if det.label in self._config.vis_class_colors:
                    color = self._config.vis_class_colors[det.label]
                else:
                    color = self._config.vis_default_detection_color
                
                # Make ground truth slightly different (darker/lighter)
                if is_ground_truth:
                    # Make GT boxes thicker and slightly darker
                    thickness = 3
                    color = tuple(int(c * 0.8) for c in color)  # Darken by 20%
                else:
                    thickness = 2
                
                cv2.rectangle(output_image, (x1, y1), (x2, y2), color, thickness)
                
                # Prepare label
                if is_ground_truth:
                    label = f"GT: {det.label}"
                else:
                    label = f"{det.label}: {det.conf:.2f}"
                
                label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                
                # Background for text - use the same color as box but darker
                bg_color = tuple(int(c * 0.5) for c in color)
                cv2.rectangle(output_image, (x1, y1 - label_size[1] - 4), 
                            (x1 + label_size[0], y1), bg_color, -1)
                
                # Text in white
                cv2.putText(output_image, label, (x1, y1 - 2), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return output_image

    def stop(self) -> None:
        """Clean up resources"""
        self.logger.info(f"Stopping {self.name}")
        if self.frame_count > 0 and not hasattr(self, '_finalized'):
            self._finalize_evaluation()
            self._finalized = True
        self.is_initialized = False


        