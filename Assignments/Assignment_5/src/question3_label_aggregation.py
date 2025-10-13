"""
Question 3: Snorkel Label Aggregation - Majority Label Voter
============================================================

WHAT: Implement Snorkel's Label aggregation using Majority Label Voter
WHY: Multiple labeling functions may disagree; we need to aggregate their votes intelligently
HOW: Use Snorkel's LabelModel with MajorityLabelVoter to combine noisy labels

Key Concepts:
- Majority Voting: Simple strategy where the most common label wins
- Label Model: Sophisticated probabilistic model that learns LF accuracies and correlations
- Weak Supervision: Converting programmatic labels into training data
- Label Aggregation: Combining multiple noisy labeling functions into clean labels

Process:
1. Load label matrix from Question 2 
2. Apply Majority Label Voter
3. Train Snorkel LabelModel 
4. Compare aggregation strategies
5. Generate final probabilistic labels for training
"""

import wandb
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from snorkel.labeling.model import MajorityLabelVoter, LabelModel
# from snorkel.analysis import LFAnalysis # Not available in current version
from snorkel.utils import probs_to_preds
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from typing import Dict, Tuple, Optional
import os
import pickle

# Constants from Question 2
ABSTAIN = -1
MISC = 0
NOT_ENTITY = 1

class SnorkelLabelAggregation:
	"""Implements Snorkel label aggregation strategies"""

	def __init__(self):
		self.label_names = ['MISC', 'NOT_ENTITY']
		self.wandb_run = None

	def initialize_wandb(self):
		"""Initialize W&B for tracking aggregation experiments"""
		print(" Initializing W&B for Question 3...")

		self.wandb_run = wandb.init(
			project="Q1-weak-supervision-ner",
			name="snorkel-label-aggregation",
			tags=["snorkel", "label-aggregation", "majority-voting"],
			notes="Question 3: Implementation of Snorkel Label Aggregation with Majority Voter"
		)

		print(" W&B initialized for label aggregation!")

	def load_question2_outputs(self) -> Tuple[np.ndarray, pd.DataFrame]:
		"""Load label matrix and data from Question 2"""
		print(" Loading outputs from Question 2...")

		try:
			# Load label matrix
			L_train = np.load('outputs/label_matrix_Q2.npy')

			# Load dataframe
			df = pd.read_pickle('outputs/dataframe_Q2.pkl')

			print(f" Loaded label matrix: {L_train.shape}")
			print(f" Loaded dataframe: {len(df)} samples")

			return L_train, df

		except FileNotFoundError as e:
			print(f" Error: Could not find Question 2 outputs. Please run Question 2 first.")
			print(f" Expected files: outputs/label_matrix_Q2.npy, outputs/dataframe_Q2.pkl")
			raise

	def analyze_label_matrix(self, L_train: np.ndarray) -> Dict:
		"""Analyze the label matrix before aggregation"""
		print(" Analyzing label matrix...")

		analysis = {}

		# Basic statistics
		analysis['shape'] = L_train.shape
		analysis['total_labels'] = np.sum(L_train != ABSTAIN)
		analysis['abstain_rate'] = np.sum(L_train == ABSTAIN) / L_train.size

		# Per-function statistics
		analysis['per_function'] = {}
		for i in range(L_train.shape[1]):
			lf_labels = L_train[:, i]
			analysis['per_function'][f'LF_{i}'] = {
				'coverage': np.sum(lf_labels != ABSTAIN) / len(lf_labels),
				'label_counts': dict(zip(*np.unique(lf_labels, return_counts=True)))
			}

		# Agreement analysis
		# Count samples where multiple LFs agree
		non_abstain_mask = L_train != ABSTAIN
		agreement_counts = {}

		for i in range(L_train.shape[0]):
			sample_labels = L_train[i, non_abstain_mask[i]]
			if len(sample_labels) > 1:
				# Check if all non-abstain labels agree
				if len(set(sample_labels)) == 1:
					num_voters = len(sample_labels)
					agreement_counts[num_voters] = agreement_counts.get(num_voters, 0) + 1

		analysis['agreement'] = agreement_counts

		print(f" Matrix shape: {analysis['shape']}")
		print(f" Total labels: {analysis['total_labels']:,}")
		print(f" Abstain rate: {analysis['abstain_rate']:.1%}")

		return analysis

	def apply_majority_label_voter(self, L_train: np.ndarray) -> Tuple[np.ndarray, Dict]:
		"""Apply Snorkel's Majority Label Voter"""
		print(" Applying Majority Label Voter...")

		# Initialize Majority Label Voter
		majority_model = MajorityLabelVoter(cardinality=2)  # 2 classes: MISC, NOT_ENTITY

		# Apply directly (no fit needed in current version)
		majority_preds = majority_model.predict(L_train)

		# Calculate coverage (non-abstain predictions)
		coverage = np.sum(majority_preds != ABSTAIN) / len(majority_preds)

		# Get prediction statistics
		unique_labels, counts = np.unique(majority_preds, return_counts=True)
		pred_distribution = dict(zip(unique_labels, counts))

		stats = {
			'coverage': coverage,
			'predictions': len(majority_preds),
			'abstain_count': np.sum(majority_preds == ABSTAIN),
			'prediction_distribution': pred_distribution
		}

		print(f" Majority voting completed!")
		print(f" Coverage: {coverage:.1%}")
		print(f" Predictions: {pred_distribution}")

		return majority_preds, stats

	def train_label_model(self, L_train: np.ndarray) -> Tuple[np.ndarray, np.ndarray, Dict]:
		"""Train Snorkel's probabilistic Label Model"""
		print(" Training Snorkel Label Model...")

		# Initialize Label Model
		label_model = LabelModel(cardinality=2, verbose=True)

		# Train the model
		label_model.fit(L_train, n_epochs=500, log_freq=100)

		# Get probabilistic predictions
		probs = label_model.predict_proba(L_train)

		# Convert probabilities to hard predictions
		preds = probs_to_preds(probs)

		# Calculate statistics
		coverage = np.sum(preds != ABSTAIN) / len(preds)
		unique_labels, counts = np.unique(preds, return_counts=True)
		pred_distribution = dict(zip(unique_labels, counts))

		# Model confidence statistics
		max_probs = np.max(probs, axis=1)
		avg_confidence = np.mean(max_probs[preds != ABSTAIN])

		stats = {
			'coverage': coverage,
			'prediction_distribution': pred_distribution,
			'average_confidence': avg_confidence,
			'model_weights': label_model.get_weights().tolist() if hasattr(label_model, 'get_weights') else None
		}

		print(f" Label Model training completed!")
		print(f" Coverage: {coverage:.1%}")
		print(f" Average confidence: {avg_confidence:.3f}")

		return preds, probs, stats

	def compare_aggregation_methods(self, df: pd.DataFrame, majority_preds: np.ndarray, model_preds: np.ndarray) -> Dict:
		"""Compare different aggregation methods against ground truth"""
		print(" Comparing aggregation methods...")

		ground_truth = df['ground_truth'].values

		comparison = {}

		# Majority Label Voter evaluation
		majority_mask = majority_preds != ABSTAIN
		if np.sum(majority_mask) > 0:
			majority_accuracy = accuracy_score(
				ground_truth[majority_mask],
				majority_preds[majority_mask]
			)

			comparison['majority_voter'] = {
				'accuracy': majority_accuracy,
				'coverage': np.sum(majority_mask) / len(majority_preds),
				'predictions_count': np.sum(majority_mask)
			}

		# Label Model evaluation
		model_mask = model_preds != ABSTAIN
		if np.sum(model_mask) > 0:
			model_accuracy = accuracy_score(
				ground_truth[model_mask],
				model_preds[model_mask]
			)

			comparison['label_model'] = {
				'accuracy': model_accuracy,
				'coverage': np.sum(model_mask) / len(model_preds),
				'predictions_count': np.sum(model_mask)
			}

		print(f" Majority Voter - Accuracy: {comparison.get('majority_voter', {}).get('accuracy', 0):.3f}")
		print(f" Label Model - Accuracy: {comparison.get('label_model', {}).get('accuracy', 0):.3f}")

		return comparison

	def create_visualizations(self, L_train: np.ndarray, majority_preds: np.ndarray, model_preds: np.ndarray, probs: np.ndarray, comparison: Dict):
		"""Create comprehensive visualizations for label aggregation"""
		print(" Creating label aggregation visualizations...")

		os.makedirs("outputs", exist_ok=True)

		fig, axes = plt.subplots(2, 3, figsize=(18, 10))

		# 1. Label Matrix Heatmap
		ax1 = axes[0, 0]
		sample_size = min(200, L_train.shape[0])
		sns.heatmap(L_train[:sample_size].T,
				ax=ax1, cmap='RdYlBu', center=0,
				yticklabels=['Year Detector', 'Org Detector', 'MISC Keywords'])
		ax1.set_title('Label Matrix (Sample)')
		ax1.set_xlabel('Samples')

		# 2. Prediction Comparison
		ax2 = axes[0, 1]
		methods = []
		accuracies = []
		coverages = []

		if 'majority_voter' in comparison:
			methods.append('Majority\nVoter')
			accuracies.append(comparison['majority_voter']['accuracy'])
			coverages.append(comparison['majority_voter']['coverage'])

		if 'label_model' in comparison:
			methods.append('Label\nModel')
			accuracies.append(comparison['label_model']['accuracy'])
			coverages.append(comparison['label_model']['coverage'])

		x_pos = np.arange(len(methods))
		ax2.bar(x_pos - 0.2, accuracies, 0.4, label='Accuracy', alpha=0.7, color='lightblue')
		ax2.bar(x_pos + 0.2, coverages, 0.4, label='Coverage', alpha=0.7, color='lightcoral')
		ax2.set_xlabel('Aggregation Method')
		ax2.set_ylabel('Score')
		ax2.set_title('Aggregation Method Comparison')
		ax2.set_xticks(x_pos)
		ax2.set_xticklabels(methods)
		ax2.legend()
		ax2.grid(True, alpha=0.3)

		# 3. Confidence Distribution (Label Model)
		ax3 = axes[0, 2]
		if probs is not None:
			max_probs = np.max(probs, axis=1)
			non_abstain_mask = model_preds != ABSTAIN

			ax3.hist(max_probs[non_abstain_mask], bins=30, alpha=0.7, color='steelblue')
			ax3.axvline(np.mean(max_probs[non_abstain_mask]), color='red', linestyle='--',
					 label=f'Mean: {np.mean(max_probs[non_abstain_mask]):.3f}')
			ax3.set_xlabel('Prediction Confidence')
			ax3.set_ylabel('Frequency')
			ax3.set_title('Label Model Confidence Distribution')
			ax3.legend()
			ax3.grid(True, alpha=0.3)

		# 4. Label Coverage by Function
		ax4 = axes[1, 0]
		lf_names = ['Year Detector', 'Org Detector', 'MISC Keywords']
		coverages_per_lf = []

		for i in range(L_train.shape[1]):
			coverage = np.sum(L_train[:, i] != ABSTAIN) / L_train.shape[0]
			coverages_per_lf.append(coverage)

		bars = ax4.bar(lf_names, coverages_per_lf, color=['skyblue', 'lightcoral', 'lightgreen'])
		ax4.set_ylabel('Coverage')
		ax4.set_title('Labeling Function Coverage')
		ax4.tick_params(axis='x', rotation=45)

		# Add value labels on bars
		for bar, coverage in zip(bars, coverages_per_lf):
			ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
					 f'{coverage:.2f}', ha='center', va='bottom')

		# 5. Prediction Agreement
		ax5 = axes[1, 1]
		if majority_preds is not None and model_preds is not None:
			# Where both methods made predictions
			both_predict_mask = (majority_preds != ABSTAIN) & (model_preds != ABSTAIN)

			if np.sum(both_predict_mask) > 0:
				agreement = majority_preds[both_predict_mask] == model_preds[both_predict_mask]
				agreement_rate = np.mean(agreement)

				ax5.pie([agreement_rate, 1-agreement_rate],
						 labels=['Agree', 'Disagree'],
						 autopct='%1.1f%%',
						 colors=['lightgreen', 'lightcoral'])
				ax5.set_title(f'Method Agreement\n({np.sum(both_predict_mask)} samples)')

		# 6. Label Distribution
		ax6 = axes[1, 2]
		all_labels = []
		method_names = []

		if majority_preds is not None:
			maj_labels = majority_preds[majority_preds != ABSTAIN]
			unique, counts = np.unique(maj_labels, return_counts=True)
			all_labels.append(counts)
			method_names.append('Majority')

		if model_preds is not None:
			model_labels = model_preds[model_preds != ABSTAIN]
			unique, counts = np.unique(model_labels, return_counts=True)
			all_labels.append(counts)
			method_names.append('Label Model')

		if len(all_labels) > 0:
			x_pos = np.arange(len(self.label_names))
			width = 0.35

			if len(all_labels) >= 1:
				ax6.bar(x_pos - width/2, all_labels[0], width,
						label=method_names[0], alpha=0.7, color='skyblue')
			if len(all_labels) >= 2:
				ax6.bar(x_pos + width/2, all_labels[1], width,
						label=method_names[1], alpha=0.7, color='lightcoral')

		ax6.set_xlabel('Label')
		ax6.set_ylabel('Count')
		ax6.set_title('Label Distribution by Method')
		ax6.set_xticks(x_pos)
		ax6.set_xticklabels(self.label_names)
		ax6.legend()

		plt.tight_layout()
		plt.savefig('outputs/question3_label_aggregation.png', dpi=300, bbox_inches='tight')
		plt.show()

	def log_to_wandb(self, analysis: Dict, majority_stats: Dict, model_stats: Dict, comparison: Dict):
		"""Log aggregation results to W&B"""
		print(" Logging label aggregation results to W&B...")

		# Log matrix analysis
		wandb.log({
			"label_matrix_shape_samples": analysis['shape'][0],
			"label_matrix_shape_functions": analysis['shape'][1],
			"label_matrix_abstain_rate": analysis['abstain_rate'],
			"total_label_applications": analysis['total_labels']
		})

		# Log majority voter results
		wandb.log({
			"majority_voter_coverage": majority_stats['coverage'],
			"majority_voter_abstain_count": majority_stats['abstain_count']
		})

		# Log label model results
		wandb.log({
			"label_model_coverage": model_stats['coverage'],
			"label_model_avg_confidence": model_stats['average_confidence']
		})

		# Log comparison metrics
		if 'majority_voter' in comparison:
			wandb.log({
				"majority_voter_accuracy": comparison['majority_voter']['accuracy'],
				"majority_voter_predictions": comparison['majority_voter']['predictions_count']
			})

		if 'label_model' in comparison:
			wandb.log({
				"label_model_accuracy": comparison['label_model']['accuracy'],
				"label_model_predictions": comparison['label_model']['predictions_count']
			})

		# Log visualization
		wandb.log({
			"label_aggregation_analysis": wandb.Image("outputs/question3_label_aggregation.png")
		})

		# Create comparison table
		aggregation_table = wandb.Table(columns=["Method", "Accuracy", "Coverage", "Predictions"])

		for method_name, metrics in comparison.items():
			aggregation_table.add_data(
				method_name.replace('_', ' ').title(),
				f"{metrics['accuracy']:.3f}",
				f"{metrics['coverage']:.3f}",
				metrics['predictions_count']
			)

		wandb.log({"aggregation_comparison": aggregation_table})

		print(" All aggregation metrics logged to W&B!")

	def save_final_labels(self, model_preds: np.ndarray, probs: np.ndarray, df: pd.DataFrame):
		"""Save final aggregated labels for downstream use"""
		print(" Saving final aggregated labels...")

		# Add predictions to dataframe
		df_with_labels = df.copy()
		df_with_labels['aggregated_labels'] = model_preds

		# Add probabilities
		df_with_labels['label_probabilities'] = [prob.tolist() for prob in probs]
		df_with_labels['confidence'] = np.max(probs, axis=1)

		# Save results
		df_with_labels.to_pickle('outputs/final_aggregated_labels_Q3.pkl')
		np.save('outputs/final_predictions_Q3.npy', model_preds)
		np.save('outputs/final_probabilities_Q3.npy', probs)

		print(" Final labels saved to outputs/")

	def print_summary(self, analysis: Dict, comparison: Dict):
		"""Print comprehensive summary of label aggregation"""
		print("\n" + "="*60)
		print(" LABEL AGGREGATION SUMMARY")
		print("="*60)

		print(f"\n Label Matrix Analysis:")
		print(f" • Shape: {analysis['shape']}")
		print(f" • Total labels: {analysis['total_labels']:,}")
		print(f" • Abstain rate: {analysis['abstain_rate']:.1%}")

		print(f"\n Aggregation Methods:")

		for method_name, metrics in comparison.items():
			print(f"\n {method_name.replace('_', ' ').title()}:")
			print(f" • Accuracy: {metrics['accuracy']:.3f}")
			print(f" • Coverage: {metrics['coverage']:.1%}")
			print(f" • Predictions: {metrics['predictions_count']:,}")

		print(f"\n Question 3 completed! Labels aggregated and ready for training.")

def main():
    """Main function for Question 3: Label Aggregation"""
    print(" Starting Question 3: Snorkel Label Aggregation")
    print("="*60)

    # Initialize aggregation class
    aggregator = SnorkelLabelAggregation()

    try:
        # Step 1: Initialize W&B
        aggregator.initialize_wandb()

        # Step 2: Load Question 2 outputs
        L_train, df = aggregator.load_question2_outputs()

        # Step 3: Analyze label matrix
        analysis = aggregator.analyze_label_matrix(L_train)

        # Step 4: Apply Majority Label Voter
        majority_preds, majority_stats = aggregator.apply_majority_label_voter(L_train)

        # Step 5: Train Label Model
        model_preds, probs, model_stats = aggregator.train_label_model(L_train)

        # Step 6: Compare methods
        comparison = aggregator.compare_aggregation_methods(df, majority_preds, model_preds)

        # Step 7: Create visualizations
        aggregator.create_visualizations(L_train, majority_preds, model_preds, probs, comparison)

        # Step 8: Log to W&B
        aggregator.log_to_wandb(analysis, majority_stats, model_stats, comparison)

        # Step 9: Save final results
        aggregator.save_final_labels(model_preds, probs, df)

        # Step 10: Print summary
        aggregator.print_summary(analysis, comparison)

        print(f"\n Question 3 completed successfully!")
        print(f" View W&B dashboard: https://wandb.ai/{wandb.run.entity}/{wandb.run.project}")

    except Exception as e:
        print(f" Error in Question 3: {e}")
        raise

    finally:
        # Finish W&B run
        wandb.finish()

if __name__ == "__main__":
    main()