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
from scipy.interpolate import interp1d


class EvaluationModule(SystemModule):
    """
    Collects and evaluates detection results
    Saves results, visualizations, and annotated images
    """
    def __init__(self, 
                 config: Config, 
                 module_name: str = "EvaluationModule",
                 yolo_mode: str = None,
                 conf_thresh: float = 0.5,
                 use_tracking: bool = False):
        """Initialize module"""
        super().__init__(config, module_name)
        self.logger = logging.getLogger(self.name)

        timestamp = datetime.now().strftime("%m%d_%H%M%S")
        tracking_str = "_tracking" if use_tracking else ""
        if yolo_mode:
            self.output_dir = os.path.join(self._config.eval_output_folder, f"{timestamp}_{yolo_mode}_{conf_thresh}{tracking_str}")
        else:
            self.output_dir = os.path.join(self._config.eval_output_folder, f"{timestamp}{tracking_str}")

        self.results = defaultdict(lambda: defaultdict(list))
        self.ground_truth_data = []
        self.image_paths = []
        self._process_data = None
        self.treshold_metrics = defaultdict(lambda: defaultdict(dict))

        self.iou_threshold = 0.5 
        self.confidence_thresholds = [0.1, 0.3, 0.5, 0.7, 0.9]
        self.eval_thresholds = np.arange(0.0, 1.0, 0.05)

        # Select detection sources based on whether tracking is being evaluated
        self.use_tracking = use_tracking
        if self.use_tracking:
            self.detection_sources = self._config.eval_detection_sources_tracked
            self.logger.info("Evaluation configured to include TRACKED detections.")
        else:
            self.detection_sources = self._config.eval_detection_sources
            self.logger.info("Evaluation configured for standard detections.")

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
        normalized_depth_maps = {name: dm for name, dm in depth_maps.items() if 'norm_' in name}
        display_names = {'norm_depth': 'Camera Depth (Normalized)', 'norm_midas': 'MiDaS (Normalized)', 'norm_pro': 'Depth Pro (Normalized)', 'norm_vggt': 'VGGT (Normalized)'}
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        axes = axes.ravel()
        
        for idx, (name, depth_map) in enumerate(normalized_depth_maps.items()):
            if idx < len(axes) and depth_map is not None:
                ax = axes[idx]
                valid_mask = depth_map > 0
                if np.any(valid_mask):
                    vmin, vmax = np.percentile(depth_map[valid_mask], 5), np.percentile(depth_map[valid_mask], 95)
                    im = ax.imshow(depth_map, cmap='viridis', vmin=vmin, vmax=vmax)
                    title = display_names.get(name, name.replace('_', ' ').title())
                    ax.set_title(title)
                    ax.axis('off')
        
        for idx in range(len(normalized_depth_maps), len(axes)):
            axes[idx].axis('off')
        
        plt.tight_layout()
        output_path = os.path.join(self.output_dir, "depth_analysis", f"{os.path.basename(image_name)}_depth_comparison.png")
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close() 

    def _finalize_evaluation(self):
        """Compute final metrics and save result"""
        has_ground_truth = any(self.ground_truth_data) and any(gt for gt in self.ground_truth_data)
        if not has_ground_truth:
            self.logger.warning("No ground truth data found. Skipping metric calculations.")
            self.logger.info(f"Annotated images saved to: {os.path.join(self.output_dir, 'annotated_images')}")
            
            # Create a placeholder summary file
            summary_path = os.path.join(self.output_dir, "evaluation_summary.txt")
            with open(summary_path, 'w') as f:
                f.write("EVALUATION SUMMARY\n" + "=" * 50 + "\n\n")
                f.write("Run was performed without ground truth labels.\n")
                f.write("Metric calculation was skipped.\n")
                f.write(f"Annotated images and other outputs are available in: {self.output_dir}\n")
            return
        import matplotlib
        matplotlib.use('Agg')
        self.logger.info(f"Computing evaluation metrics")

        metrics = {}

        for source in self.detection_sources:
            if source in self.results and self.results[source]['detections']:
                source_metrics = self._compute_detection_metrics(
                    self.results[source]['detections'],
                    self.ground_truth_data,
                )
                metrics[source] = source_metrics

        depth_metrics = self._compute_depth_metrics()
        metrics['depth_metrics'] = depth_metrics

        self._save_metrics(metrics)
        self._generate_plots(metrics)

        self.logger.info(f"Evaluation complete, results saved to {self.output_dir}")

    def _compute_detection_metrics(self, predictions: List[List[Detection]], ground_truths: List[List[Detection]]) -> Dict[str, Any]:
        """Compute object detection metrics"""
        all_predictions = [p for frame_preds in predictions for p in frame_preds]
        all_ground_truths = [gt for frame_gt in ground_truths for gt in frame_gt]

        metrics = {
            'total_predictions': len(all_predictions),
            'total_ground_truth': len(all_ground_truths),
            'per_class_metrics': {},
            'overall_metrics': {}
        }
        
        all_classes = set(det.class_id for det in all_predictions + all_ground_truths)
        
        for class_id in all_classes:
            class_preds = [d for d in all_predictions if d.class_id == class_id]
            class_gt = [d for d in all_ground_truths if d.class_id == class_id]

            if class_gt:
                class_metrics = self._compute_class_metrics(class_preds, class_gt)
                class_name = self._config.class_names.get(class_id, f"class_{class_id}")
                metrics['per_class_metrics'][class_name] = class_metrics

        if all_ground_truths:
            overall_metrics = self._compute_class_metrics(all_predictions, all_ground_truths)
            metrics['overall_metrics'] = overall_metrics
        
        threshold_metrics = self._compute_metrics_at_thresholds(all_predictions, all_ground_truths)
        metrics['threshold_analysis'] = threshold_metrics
        
        best_threshold, best_f1 = 0.5, 0
        for thresh_data in threshold_metrics.values():
            if thresh_data['f1_score'] > best_f1:
                best_f1, best_threshold = thresh_data['f1_score'], thresh_data['threshold']
        
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
            'mean_iou_for_tp': 0.0,  
            'precision_recall_curve': {'precisions': [], 'recalls': []},
            'confusion_matrix': {'tp': 0, 'fp': 0, 'fn': 0}
        }

        if not ground_truths:
            metrics['confusion_matrix']['fp'] = len(predictions)
            return metrics

        metrics['average_precision'] = self._compute_ap_standard(predictions, ground_truths, self.iou_threshold)

        sorted_preds = sorted(predictions, key=lambda x: x.conf, reverse=True)
        
        conf_threshold = self._config.confidence_threshold
        filtered_preds = [p for p in sorted_preds if p.conf >= conf_threshold]
        
        tp, fp, fn, tp_iou_scores = self._match_detections(filtered_preds, ground_truths)
        
        if tp_iou_scores:
            metrics['mean_iou_for_tp'] = np.mean(tp_iou_scores)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / len(ground_truths) if ground_truths else 0.0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        
        metrics['precision'] = precision
        metrics['recall'] = recall
        metrics['f1_score'] = f1_score
        metrics['confusion_matrix']['tp'] = tp
        metrics['confusion_matrix']['fp'] = fp
        metrics['confusion_matrix']['fn'] = fn
        
        tp_cumsum, fp_cumsum = np.zeros(len(sorted_preds)), np.zeros(len(sorted_preds))
        matched_gt = set()
        for i, pred in enumerate(sorted_preds):
            best_iou, best_gt_idx = -1, -1
            for gt_idx, gt in enumerate(ground_truths):
                iou = self._compute_iou(pred.bbox2D, gt.bbox2D)
                if iou > best_iou: best_iou, best_gt_idx = iou, gt_idx
            
            if best_iou >= self.iou_threshold:
                if best_gt_idx not in matched_gt:
                    tp_cumsum[i] = 1; matched_gt.add(best_gt_idx)
                else: fp_cumsum[i] = 1
            else: fp_cumsum[i] = 1
        
        tp_cumsum = np.cumsum(tp_cumsum)
        fp_cumsum = np.cumsum(fp_cumsum)
        recalls = tp_cumsum / len(ground_truths) if ground_truths else np.zeros_like(tp_cumsum)
        precisions = tp_cumsum / (tp_cumsum + fp_cumsum) if np.any(tp_cumsum + fp_cumsum) else np.zeros_like(tp_cumsum)
        
        metrics['precision_recall_curve']['precisions'] = precisions.tolist()
        metrics['precision_recall_curve']['recalls'] = recalls.tolist()

        return metrics
    
    def _match_detections(self, predictions: List[Detection], ground_truths: List[Detection]) -> Tuple[int, int, int, List[float]]:
        """Match detections to ground truth, returning TP, FP, FN and IoU scores for TPs."""
        tp = 0
        fp = 0
        matched_gt = set()
        tp_iou_scores = []

        sorted_preds = sorted(predictions, key=lambda x: x.conf, reverse=True)

        for pred in sorted_preds:
            best_iou = -1
            best_gt_idx = -1

            for gt_idx, gt in enumerate(ground_truths):
                if gt_idx in matched_gt:
                    continue
                
                iou = self._compute_iou(pred.bbox2D, gt.bbox2D)
                if iou > best_iou:
                    best_iou, best_gt_idx = iou, gt_idx

            if best_iou >= self.iou_threshold:
                if best_gt_idx not in matched_gt:
                    tp += 1
                    matched_gt.add(best_gt_idx)
                    tp_iou_scores.append(best_iou) 
                else:
                    fp += 1
            else:
                fp += 1
        
        fn = len(ground_truths) - len(matched_gt)

        return tp, fp, fn, tp_iou_scores

    def _compute_iou(self, box1: List[float], box2: List[float]) -> float:
        """Compute IoU between two bounding boxes"""
        if not box1 or not box2: return 0.0
        x1_1, y1_1, x2_1, y2_1 = box1
        x1_2, y1_2, x2_2, y2_2 = box2
        x1_i, y1_i = max(x1_1, x1_2), max(y1_1, y1_2)
        x2_i, y2_i = min(x2_1, x2_2), min(y2_1, y2_2)

        inter_area = max(0, x2_i - x1_i) * max(0, y2_i - y1_i)
        box1_area = (x2_1 - x1_1) * (y2_1 - y1_1)
        box2_area = (x2_2 - x1_2) * (y2_2 - y1_2)
        union_area = box1_area + box2_area - inter_area
        return inter_area / union_area if union_area > 0 else 0.0
    
    def _compute_ap_standard(self, predictions: List[Detection], ground_truths: List[Detection], iou_threshold: float) -> float:
        """Compute standard Average Precision (AP) using proper ranking and interpolation"""
        if not ground_truths or not predictions: return 0.0
        
        sorted_preds = sorted(predictions, key=lambda x: x.conf, reverse=True)
        
        tp = np.zeros(len(sorted_preds))
        fp = np.zeros(len(sorted_preds))
        matched_gt = set()
        
        for i, pred in enumerate(sorted_preds):
            best_iou, best_gt_idx = -1, -1
            for gt_idx, gt in enumerate(ground_truths):
                iou = self._compute_iou(pred.bbox2D, gt.bbox2D)
                if iou > best_iou:
                    best_iou, best_gt_idx = iou, gt_idx
            
            if best_iou >= iou_threshold:
                if best_gt_idx not in matched_gt:
                    tp[i] = 1
                    matched_gt.add(best_gt_idx)
                else:
                    fp[i] = 1
            else:
                fp[i] = 1
        
        tp_cumsum, fp_cumsum = np.cumsum(tp), np.cumsum(fp)
        recalls = tp_cumsum / len(ground_truths)
        precisions = tp_cumsum / (tp_cumsum + fp_cumsum)
        
        precisions = np.concatenate(([0.0], precisions, [0.0]))
        recalls = np.concatenate(([0.0], recalls, [1.0]))
        
        for i in range(len(precisions) - 2, -1, -1):
            precisions[i] = max(precisions[i], precisions[i + 1])
        
        indices = np.where(recalls[1:] != recalls[:-1])[0]
        ap = np.sum((recalls[indices + 1] - recalls[indices]) * precisions[indices + 1])
        
        return ap
    
    def _compute_metrics_at_thresholds(self, predictions: List[Detection], ground_truths: List[Detection]) -> Dict[str, Any]:
        """Compute metrics at multiple confidence thresholds"""
        threshold_results = {}
        
        for threshold in self.eval_thresholds:
            filtered_preds = [p for p in predictions if p.conf >= threshold]
            tp, fp, fn, _ = self._match_detections(filtered_preds, ground_truths)
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 1.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
            
            threshold_results[f'thresh_{threshold:.2f}'] = {
                'precision': precision, 'recall': recall, 'f1_score': f1,
                'tp': tp, 'fp': fp, 'fn': fn, 'threshold': threshold
            }
        
        return threshold_results

    def _compute_depth_metrics(self) -> Dict[str, Any]:
        """Analyze depth maps"""
        metrics = {}
        if 'depth_stats' not in self.results: return metrics
        
        aggregated_stats = defaultdict(lambda: defaultdict(list))
        for frame_stats in self.results['depth_stats'].values():
            for source, stats in frame_stats.items():
                for metric, value in stats.items():
                    aggregated_stats[source][metric].append(value)

        for source, stats in aggregated_stats.items():
            source_metrics = {}
            for metric, values in stats.items():
                if values:
                    source_metrics[f"{metric}_mean"] = np.mean(values)
                    source_metrics[f"{metric}_std"] = np.std(values)    
            metrics[source] = source_metrics
        return metrics
    
    def _save_metrics(self, metrics: Dict[str, Any]):
        """Save metrics to JSON file"""
        output_path = os.path.join(self.output_dir, "evaluation_metrics.json")
        
        def convert_types(obj):
            if isinstance(obj, np.ndarray): return obj.tolist()
            if isinstance(obj, (np.float32, np.float64)): return float(obj)
            if isinstance(obj, (np.int32, np.int64)): return int(obj)
            if isinstance(obj, dict): return {k: convert_types(v) for k, v in obj.items()}
            if isinstance(obj, list): return [convert_types(v) for v in obj]
            return obj
        
        metrics_serializable = convert_types(metrics)
        with open(output_path, 'w') as f:
            json.dump(metrics_serializable, f, indent=2)
        
        summary_path = os.path.join(self.output_dir, "evaluation_summary.txt")
        with open(summary_path, 'w') as f:
            f.write("EVALUATION SUMMARY\n" + "=" * 50 + "\n\n")
            f.write("OBJECT DETECTION METRICS\n" + "-" * 30 + "\n")
            
            for source, source_metrics in metrics.items():
                if source != 'depth_metrics' and isinstance(source_metrics, dict):
                    f.write(f"\n{source} (at IoU >= {self.iou_threshold}):\n")
                    if 'overall_metrics' in source_metrics:
                        overall = source_metrics['overall_metrics']
                        ap = overall.get('average_precision', 0)
                        prec = overall.get('precision', 0)
                        rec = overall.get('recall', 0)
                        f1 = overall.get('f1_score', 0)
                        miou_tp = overall.get('mean_iou_for_tp', 0) 

                        f.write(f"  - Average Precision: {float(ap):.3f}\n")
                        f.write(f"  - Precision: {float(prec):.3f}\n")
                        f.write(f"  - Recall: {float(rec):.3f}\n")
                        f.write(f"  - F1 Score: {float(f1):.3f}\n")
                        f.write(f"  - Mean IoU (for TPs): {float(miou_tp):.3f}\n") 
            
            if 'depth_metrics' in metrics:
                f.write("\n\nDEPTH ESTIMATION ANALYSIS\n" + "-" * 30 + "\n")
                for source, d_metrics in metrics['depth_metrics'].items():
                    if isinstance(d_metrics, dict):
                        f.write(f"\n{source}:\n")
                        for metric, value in d_metrics.items():
                            f.write(f"  {metric}: {float(value):.3f}\n")
    
    def _generate_plots(self, metrics: Dict[str, Any]):
        """Generate all evaluation plots"""
        os.makedirs(os.path.join(self.output_dir, "plots"), exist_ok=True)
        
        try: self._plot_ap_comparison(metrics)
        except Exception as e: self.logger.error(f"Error creating AP comparison plot: {e}")
        try: self._plot_pr_curves(metrics)
        except Exception as e: self.logger.error(f"Error creating PR curves: {e}")
        try: self._plot_depth_consistency(metrics)
        except Exception as e: self.logger.error(f"Error creating depth consistency plot: {e}")
        try: self._plot_per_class_performance(metrics)
        except Exception as e: self.logger.error(f"Error creating per-class performance plot: {e}")
        try: self._plot_confusion_matrix(metrics)
        except Exception as e: self.logger.error(f"Error creating confusion matrices: {e}")
        try: self._plot_metrics_summary(metrics)
        except Exception as e: self.logger.error(f"Error creating metrics summary: {e}")
        try: self._plot_threshold_analysis(metrics)
        except Exception as e: self.logger.error(f"Error creating threshold analysis plot: {e}")
    
    def _plot_metrics_summary(self, metrics: Dict[str, Any]):
        """Plot a summary of key metrics across all sources"""
        sources, precisions, recalls, f1_scores = [], [], [], []
        
        for source, source_metrics in metrics.items():
            if source != 'depth_metrics' and 'overall_metrics' in source_metrics:
                overall = source_metrics['overall_metrics']
                sources.append(source.replace('_', ' ').title())
                precisions.append(overall.get('precision', 0))
                recalls.append(overall.get('recall', 0))
                f1_scores.append(overall.get('f1_score', 0))
        
        if not sources: return
        
        x, width = np.arange(len(sources)), 0.25
        fig, ax = plt.subplots(figsize=(12, 6))
        bars1 = ax.bar(x - width, precisions, width, label='Precision', color='skyblue')
        bars2 = ax.bar(x, recalls, width, label='Recall', color='lightcoral')
        bars3 = ax.bar(x + width, f1_scores, width, label='F1 Score', color='lightgreen')
        
        ax.set_xlabel('Detection Source')
        ax.set_ylabel('Score')
        ax.set_title(f'Detection Performance Metrics Summary (IoU@{self.iou_threshold})')
        ax.set_xticks(x)
        ax.set_xticklabels(sources, rotation=45, ha='right')
        ax.legend()
        ax.set_ylim(0, 1.1)
        
        def autolabel(bars):
            for bar in bars:
                height = bar.get_height()
                ax.annotate(f'{height:.2f}', xy=(bar.get_x() + bar.get_width() / 2, height),
                            xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=8)
        
        autolabel(bars1); autolabel(bars2); autolabel(bars3)
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "plots", "metrics_summary.png"), dpi=150)
        plt.close()

    def _plot_ap_comparison(self, metrics: Dict[str, Any]):
        """Plot Average Precision comparison across sources"""
        sources, aps = [], []
        
        for source, source_metrics in metrics.items():
            if source != 'depth_analysis' and 'overall_metrics' in source_metrics:
                sources.append(source.replace('_', ' ').title())
                aps.append(source_metrics['overall_metrics'].get('average_precision', 0))
        
        if not sources: return
        
        plt.figure(figsize=(10, 6))
        bars = plt.bar(sources, aps)
        
        for bar, ap in zip(bars, aps):
            if ap > 0.7: bar.set_color('green')
            elif ap > 0.5: bar.set_color('orange')
            else: bar.set_color('red')
        
        plt.ylabel('Average Precision')
        plt.title(f'Object Detection Performance Comparison (AP @ {self.iou_threshold} IoU)')
        plt.xticks(rotation=45, ha='right')
        plt.ylim(0, 1)
        
        for bar, ap in zip(bars, aps):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{ap:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "plots", "ap_comparison.png"), dpi=150)
        plt.close()
    
    def _plot_threshold_analysis(self, metrics: Dict[str, Any]):
        """Plot precision, recall, and F1 vs confidence threshold"""
        valid_sources = [(s, sm) for s, sm in metrics.items() if s != 'depth_metrics' and 'threshold_analysis' in sm]
        if not valid_sources: return

        n_sources = len(valid_sources)
        cols, rows = min(2, n_sources), (n_sources + 1) // 2
        fig, axes = plt.subplots(rows, cols, figsize=(15, 8 * rows), squeeze=False)
        axes = axes.ravel()

        for plot_idx, (source, source_metrics) in enumerate(valid_sources):
            ax = axes[plot_idx]
            thresh_data = source_metrics['threshold_analysis']
            thresholds, precisions, recalls, f1_scores = [], [], [], []
            
            for data in sorted(thresh_data.values(), key=lambda x: x['threshold']):
                thresholds.append(data['threshold'])
                precisions.append(data['precision'])
                recalls.append(data['recall'])
                f1_scores.append(data['f1_score'])
            
            ax.plot(thresholds, precisions, 'b-', label='Precision', linewidth=2)
            ax.plot(thresholds, recalls, 'r-', label='Recall', linewidth=2)
            ax.plot(thresholds, f1_scores, 'g-', label='F1 Score', linewidth=2)
            
            optimal_thresh = source_metrics.get('optimal_threshold', 0.5)
            ax.axvline(x=optimal_thresh, color='black', linestyle='--', alpha=0.7, label=f'Optimal ({optimal_thresh:.2f})')
            ax.axvline(x=self._config.confidence_threshold, color='red', linestyle=':', alpha=0.7, label=f'Current ({self._config.confidence_threshold:.2f})')
            
            ax.set_xlabel('Confidence Threshold'); ax.set_ylabel('Score')
            ax.set_title(f'{source.replace("_", " ").title()}')
            ax.legend(); ax.grid(True, alpha=0.3); ax.set_xlim(0, 1); ax.set_ylim(0, 1.1)
        
        for idx in range(n_sources, len(axes)):
            axes[idx].axis('off')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "plots", "threshold_analysis.png"), dpi=150)
        plt.close()

    def _plot_pr_curves(self, metrics: Dict[str, Any]):
        """Plot Precision-Recall curves for each detection source"""
        plt.figure(figsize=(12, 8))
        color_map = {'color': "#F14B4B", 'depth': "#9F33D1", 'midas': "#407FBE", 'pro': "#349649", 'vggt': "#FCEA4C"}
        
        def get_color_key(s_name: str) -> str:
            s = s_name.lower()
            if 'midas' in s: return 'midas'
            if 'pro' in s: return 'pro'
            if 'vggt' in s: return 'vggt'
            if 'depth' in s: return 'depth'
            return 'color'

        for plot_idx, (source, source_metrics) in enumerate(metrics.items()):
            if source != 'depth_metrics' and 'overall_metrics' in source_metrics:
                overall = source_metrics['overall_metrics']
                if 'precision_recall_curve' in overall:
                    pr_data = overall['precision_recall_curve']
                    precisions = pr_data.get('precisions', [])
                    recalls = pr_data.get('recalls', [])
                    
                    if recalls and precisions:
                        ap = overall.get('average_precision', 0)
                        is_tracked = 'tracked_' in source
                        color = color_map.get(get_color_key(source), f'C{plot_idx}')
                        linestyle = '--' if is_tracked else '-'
                        source_name = source.replace('_detections', '').replace('_', ' ').title()
                        
                        plt.plot(recalls, precisions, color=color, linestyle=linestyle, label=f'{source_name} (AP={ap:.3f})', linewidth=2.5)
        
        plt.xlabel('Recall', fontsize=12); plt.ylabel('Precision', fontsize=12)
        plt.title(f'Precision-Recall Curves (IoU@{self.iou_threshold})', fontsize=14)
        plt.legend(loc='lower left', fontsize=10); plt.grid(True, alpha=0.3)
        plt.xlim([0, 1]); plt.ylim([0, 1.05])
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "plots", "precision_recall_curves.png"), dpi=150, bbox_inches='tight')
        plt.close()

    def _plot_depth_consistency(self, metrics: Dict[str, Any]):
        """Plot depth estimation consistency heatmap"""
        if 'depth_metrics' not in metrics: return
        
        depth_sources = ['depth', 'midas_estimated_depth', 'pro_estimated_depth', 'vggt_estimated_depth']
        display_names = ['Camera', 'MiDaS', 'Depth Pro', 'VGGT']
        
        n_sources = len(depth_sources)
        consistency_matrix = np.ones((n_sources, n_sources))
        depth_metrics = metrics['depth_metrics']
        
        for i, source1 in enumerate(depth_sources):
            for j, source2 in enumerate(depth_sources):
                if i != j and source1 in depth_metrics and source2 in depth_metrics:
                    valid1 = depth_metrics[source1].get('valid_ratio_mean', 0)
                    valid2 = depth_metrics[source2].get('valid_ratio_mean', 0)
                    consistency_matrix[i, j] = 1 - abs(valid1 - valid2)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(consistency_matrix, xticklabels=display_names, yticklabels=display_names,
                annot=True, fmt='.3f', cmap='RdYlGn', vmin=0, vmax=1,
                cbar_kws={'label': 'Consistency Score'})
        
        plt.title('Depth Estimation Consistency')
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "plots", "depth_consistency.png"), dpi=150)
        plt.close()

    def _plot_confusion_matrix(self, metrics: Dict[str, Any]):
        """Plot separate confusion matrices for each detection source"""
        valid_sources = [(s, sm) for s, sm in metrics.items() if s != 'depth_metrics' and 'per_class_metrics' in sm]
        if not valid_sources: return
        
        class_names = list(self._config.class_names.values())
        n_classes = len(class_names)
        
        for source, source_metrics in valid_sources:
            source_name = source.replace('_detections', '').replace('_', ' ').title()
            conf_matrix = np.zeros((n_classes, n_classes))
            
            for i, class_name in enumerate(class_names):
                if class_name in source_metrics['per_class_metrics']:
                    cm = source_metrics['per_class_metrics'][class_name].get('confusion_matrix', {})
                    tp, fp, fn = cm.get('tp', 0), cm.get('fp', 0), cm.get('fn', 0)
                    conf_matrix[i, i] = tp
                    if n_classes > 1:
                        for j in range(n_classes):
                            if i != j:
                                conf_matrix[j, i] += fp / (n_classes - 1)
                                conf_matrix[i, j] += fn / (n_classes - 1)
            
            fig, ax = plt.subplots(figsize=(8, 6))
            row_sums = conf_matrix.sum(axis=1, keepdims=True)
            row_sums[row_sums == 0] = 1
            conf_matrix_norm = conf_matrix / row_sums * 100
            
            im = ax.imshow(conf_matrix_norm, cmap='Blues', vmin=0, vmax=100)
            ax.set_xticks(range(n_classes)); ax.set_yticks(range(n_classes))
            ax.set_xticklabels(class_names, rotation=45, ha='right'); ax.set_yticklabels(class_names)
            ax.set_xlabel('Predicted', fontsize=12); ax.set_ylabel('Actual', fontsize=12)
            ax.set_title(f'Confusion Matrix - {source_name} (IoU@{self.iou_threshold})', fontsize=14)
            
            for i in range(n_classes):
                for j in range(n_classes):
                    value = conf_matrix_norm[i, j]
                    ax.text(j, i, f'{value:.1f}%', ha='center', va='center', color='white' if value > 50 else 'black', fontsize=10)
            
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label='Percentage (%)')
            plt.tight_layout()
            safe_source_name = source.replace('_detections', '').replace('_', '-')
            plt.savefig(os.path.join(self.output_dir, "plots", f"confusion_matrix_{safe_source_name}.png"), dpi=150, bbox_inches='tight')
            plt.close()
        
    def _plot_per_class_performance(self, metrics: Dict[str, Any]):
        """Plot per-class performance comparison"""
        class_names = list(self._config.class_names.values())
        sources, class_aps = [], defaultdict(list)
        
        for source, source_metrics in metrics.items():
            if source != 'depth_analysis' and 'per_class_metrics' in source_metrics:
                sources.append(source.replace('_', ' ').title())
                for class_name in class_names:
                    ap = source_metrics['per_class_metrics'].get(class_name, {}).get('average_precision', 0)
                    class_aps[class_name].append(ap)
        
        if not sources or not class_aps: return
        
        x = np.arange(len(sources))
        width = 0.8 / len(class_names)
        fig, ax = plt.subplots(figsize=(12, 6))
        
        for i, class_name in enumerate(class_names):
            offset = width * (i - len(class_names)/2 + 0.5)
            ax.bar(x + offset, class_aps[class_name], width, label=class_name)
        
        ax.set_xlabel('Detection Source'); ax.set_ylabel('Average Precision')
        ax.set_title(f'Per-Class Detection Performance (IoU@{self.iou_threshold})')
        ax.set_xticks(x); ax.set_xticklabels(sources, rotation=45, ha='right')
        ax.legend(); ax.set_ylim(0, 1)
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "plots", "per_class_performance.png"), dpi=150)
        plt.close()
        
    def _save_annotated_image(self, image: np.ndarray, ground_truth: List[Detection], 
                        detections: Dict[str, List[Detection]], image_name: str,
                        current_data: Dict[str, Any]):
        """Save image with ground truth and detection overlays"""
        import cv2
        
        subdirs = ['color_with_gt', 'color_with_detections', 'normalized_depths', 'depth_with_detections', 'depth_with_gt', 'montages', 'comparisons']
        for subdir in subdirs:
            os.makedirs(os.path.join(self.output_dir, "annotated_images", subdir), exist_ok=True)
        
        normalized_depths = {}
        norm_mapping = { SystemData.NORM_DEPTH: 'depth', SystemData.NORM_MIDAS: 'midas', SystemData.NORM_PRO: 'depthpro', SystemData.NORM_VGGT: 'vggt' }
        
        for norm_key, name in norm_mapping.items():
            if norm_key in current_data and current_data[norm_key] is not None:
                normalized_depths[name] = current_data[norm_key]
        
        base_name = os.path.splitext(os.path.basename(image_name))[0]
        
        def put_text_with_background(img, text, font_scale=0.7, thickness=2):
            font = cv2.FONT_HERSHEY_SIMPLEX
            text_size, _ = cv2.getTextSize(text, font, font_scale, thickness)
            text_w, text_h = text_size
            y = text_h + 5
            cv2.rectangle(img, (0, 0), (text_w + 10, y + 5), (0, 0, 0), -1)
            cv2.putText(img, text, (5, y), font, font_scale, (255, 255, 255), thickness)
            return img
        
        gt_color_image = self._draw_detections_with_colors(image.copy(), ground_truth, is_ground_truth=True)
        gt_color_image = put_text_with_background(gt_color_image, "Ground Truth")
        cv2.imwrite(os.path.join(self.output_dir, "annotated_images", "color_with_gt", f"{base_name}.png"), gt_color_image)
        
        for source, dets in detections.items():
            if dets:
                color_det_image = self._draw_detections_with_colors(image.copy(), dets, is_ground_truth=False)
                source_name = source.replace("_detections", "").replace("_", " ").title()
                color_det_image = put_text_with_background(color_det_image, source_name)
                cv2.imwrite(os.path.join(self.output_dir, "annotated_images", "color_with_detections", f"{base_name}_{source}.png"), color_det_image)

        source_to_depth_map = {
            SystemData.DEPTH_DETECTIONS: 'depth',
            SystemData.MIDAS_DETECTIONS: 'midas',
            SystemData.PRO_DETECTIONS: 'depthpro',
            SystemData.VGGT_DETECTIONS: 'vggt',
            SystemData.TRACKED_DEPTH_DETECTIONS: 'depth',
            SystemData.TRACKED_MIDAS_DETECTIONS: 'midas',
            SystemData.TRACKED_PRO_DETECTIONS: 'depthpro',
            SystemData.TRACKED_VGGT_DETECTIONS: 'vggt',
        }

        if 'depth' in normalized_depths:
            depth_map_gray = normalized_depths['depth']
            depth_map_bgr = cv2.cvtColor(depth_map_gray, cv2.COLOR_GRAY2BGR)
            gt_depth_image = self._draw_detections_with_colors(depth_map_bgr, ground_truth, is_ground_truth=True)
            gt_depth_image = put_text_with_background(gt_depth_image, "Ground Truth on Depth")
            cv2.imwrite(os.path.join(self.output_dir, "annotated_images", "depth_with_gt", f"{base_name}.png"), gt_depth_image)

        for source, dets in detections.items():
            if not dets:
                continue

            depth_key = source_to_depth_map.get(source)
            if depth_key and depth_key in normalized_depths:
                depth_map_gray = normalized_depths[depth_key]
                
                depth_map_bgr = cv2.cvtColor(depth_map_gray, cv2.COLOR_GRAY2BGR)
                
                depth_det_image = self._draw_detections_with_colors(depth_map_bgr, dets, is_ground_truth=False)
                
                source_name = source.replace("_detections", " detections").replace("_", " ").title()
                depth_det_image = put_text_with_background(depth_det_image, source_name)
                
                output_filename = os.path.join(self.output_dir, "annotated_images", "depth_with_detections", f"{base_name}_{source}.png")
                cv2.imwrite(output_filename, depth_det_image)

    def _draw_detections_with_colors(self, image: np.ndarray, detections: List[Detection], is_ground_truth: bool = False) -> np.ndarray:
        """Draw detections with class-specific colors from config"""
        import cv2
        output_image = image.copy()
        for det in detections:
            if det.conf > 0.6 and det.bbox2D:
                x1, y1, x2, y2 = [int(v) for v in det.bbox2D]
                color = self._config.vis_class_colors.get(det.label, self._config.vis_default_detection_color)
                thickness = 3 if is_ground_truth else 2
                if is_ground_truth: color = tuple(int(c * 0.8) for c in color)
                cv2.rectangle(output_image, (x1, y1), (x2, y2), color, thickness)
                label = f"GT: {det.label}" if is_ground_truth else f"{det.label}: {det.conf:.2f}"
                if not is_ground_truth and det.track_id is not None:
                    label += f" | ID: {det.track_id}"
                label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                bg_color = tuple(int(c * 0.5) for c in color)
                cv2.rectangle(output_image, (x1, y1 - label_size[1] - 4), (x1 + label_size[0], y1), bg_color, -1)
                cv2.putText(output_image, label, (x1, y1 - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        return output_image

    def stop(self) -> None:
        """Clean up resources"""
        self.logger.info(f"Stopping {self.name}")
        if self.frame_count > 0 and not hasattr(self, '_finalized'):
            self._finalize_evaluation()
            self._finalized = True
        self.is_initialized = False