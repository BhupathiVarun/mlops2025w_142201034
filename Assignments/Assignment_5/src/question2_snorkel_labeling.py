"""
Question 2: Snorkel AI Labeling Functions for NER
=================================================

WHAT: Implement two labeling functions using Snorkel AI for weak supervision
WHY: Manual labeling is expensive; programmatic labeling functions can create training data automatically
HOW: Create heuristic rules to identify entities and evaluate their performance

Labeling Functions:
1. Year Detection: Identify years (1900-2099) as potential DATE/MISC entities
2. Organization Detection: Identify organizations by common suffixes (Inc., Corp., Ltd.)

Weak Supervision Concept:
- Instead of manually labeling thousands of examples, we write functions that automatically label data
- These functions capture domain knowledge as programmatic rules
- Multiple functions can vote on the same data point
- Snorkel aggregates these votes to create probabilistic labels
"""

import wandb
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datasets import load_dataset
from snorkel.labeling import labeling_function, PandasLFApplier
from snorkel.labeling.model import LabelModel
import re
from typing import List, Dict, Tuple, Optional
from collections import Counter, defaultdict
import os

# Constants for labeling
ABSTAIN = -1  # Snorkel constant for "no label" 
MISC = 0     # Our label for MISC entities
NOT_ENTITY = 1  # Our label for non-entities

class SnorkelLabelingFunctions:
    """Implements Snorkel labeling functions for NER weak supervision"""
    
    def __init__(self):
        self.label_names = ['MISC', 'NOT_ENTITY']
        self.wandb_run = None
        
    def initialize_wandb(self):
        """Initialize W&B for logging labeling function performance"""
        print(" Initializing W&B for Question 2...")
        
        self.wandb_run = wandb.init(
            project="Q1-weak-supervision-ner",
            name="snorkel-labeling-functions", 
            tags=["snorkel", "weak-supervision", "labeling-functions"],
            notes="Question 2: Implementation of Snorkel labeling functions for year and organization detection"
        )
        print(" W&B initialized for labeling functions!")

    def load_conll_data(self):
        """Load and prepare CoNLL-2003 data for labeling functions"""
        print(" Loading CoNLL-2003 dataset...")
        
        # Use the same robust loading approach as Question 1
        dataset_identifiers = [
            "conll2003",  # Standard identifier
            "eriktks/conll2003",  # Original attempt
            "nielsr/conll2003-ner",  # Alternative format
            "tner/conll2003",  # Another alternative
        ]
        
        dataset = None
        for i, dataset_id in enumerate(dataset_identifiers):
            try:
                print(f" Attempting to load dataset: '{dataset_id}'...")
                
                if dataset_id == "conll2003":
                    # Try the standard conll2003 dataset with trust_remote_code
                    dataset = load_dataset(dataset_id, trust_remote_code=True)
                else:
                    # Try other alternatives
                    dataset = load_dataset(dataset_id)
                
                print(f" Dataset loaded successfully using '{dataset_id}'!")
                
                # Verify the dataset has the expected structure
                sample = dataset['train'][0]
                if 'tokens' in sample and ('ner_tags' in sample or 'tags' in sample):
                    print(" Dataset structure verified!")
                    break
                else:
                    print(f" Dataset structure mismatch. Available keys: {list(sample.keys())}")
                    continue
                    
            except Exception as e:
                print(f" Failed to load '{dataset_id}': {e}")
                if i < len(dataset_identifiers) - 1:
                    print(" Trying next alternative...")
                    continue
                else:
                    raise Exception("Could not load CoNLL-2003 dataset from any source")
        
        if dataset is None:
            raise Exception("Could not load CoNLL-2003 dataset from any source")
        
        # Auto-detect field names (similar to Question 1)
        sample = dataset['train'][0]
        tokens_field = 'tokens' if 'tokens' in sample else 'words'
        labels_field = 'ner_tags' if 'ner_tags' in sample else 'tags'
        
        print(f" Using tokens field: '{tokens_field}'")
        print(f" Using labels field: '{labels_field}'")
        
        # Convert to format suitable for labeling functions
        samples = []
        
        # Use train split for demonstration
        for sample in dataset['train']:
            tokens = sample[tokens_field]
            ner_tags = sample[labels_field]
            
            # Join tokens into sentence for pattern matching
            sentence = ' '.join(tokens)
            
            # Create ground truth labels for evaluation
            # Check if sentence contains MISC entities (label 7 or 8)
            has_misc = any(tag in [7, 8] for tag in ner_tags)  # B-MISC or I-MISC
            ground_truth = MISC if has_misc else NOT_ENTITY
            
            samples.append({
                'sentence': sentence,
                'tokens': tokens,
                'ner_tags': ner_tags,
                'ground_truth': ground_truth
            })
        
        # Convert to DataFrame for Snorkel
        df = pd.DataFrame(samples)
        print(f" Loaded {len(df)} samples for labeling")
        
        return df

    def get_labeling_functions(self):
        """Get all labeling functions for this class"""
        return [lf_year_detector, lf_organization_detector, lf_misc_keywords]

    def apply_labeling_functions(self, df: pd.DataFrame) -> np.ndarray:
        """Apply all labeling functions to the dataset"""
        print(" Applying labeling functions...")
        
        # Get all labeling functions
        lfs = [
            lf_year_detector,
            lf_organization_detector, 
            lf_misc_keywords
        ]
        
        # Apply labeling functions using Snorkel
        applier = PandasLFApplier(lfs=lfs)
        L_train = applier.apply(df=df)
        
        print(f" Applied {len(lfs)} labeling functions to {len(df)} samples")
        print(f"   Label matrix shape: {L_train.shape}")
        
        return L_train, lfs

    def evaluate_labeling_functions(self, df: pd.DataFrame, L_train: np.ndarray, lfs: List) -> Dict:
        """Evaluate coverage and accuracy of labeling functions"""
        print(" Evaluating labeling function performance...")
        
        results = {}
        
        for i, lf in enumerate(lfs):
            lf_name = lf.name
            lf_labels = L_train[:, i]
            
            # Calculate coverage (% of non-abstain labels)
            coverage = np.sum(lf_labels != ABSTAIN) / len(lf_labels)
            
            # Calculate accuracy on labeled examples
            # Compare with ground truth where LF didn't abstain
            labeled_mask = lf_labels != ABSTAIN
            
            if np.sum(labeled_mask) > 0:
                labeled_predictions = lf_labels[labeled_mask]
                labeled_ground_truth = df.loc[labeled_mask, 'ground_truth'].values
                
                accuracy = np.sum(labeled_predictions == labeled_ground_truth) / len(labeled_predictions)
            else:
                accuracy = 0.0
            
            # Label distribution
            label_counts = Counter(lf_labels[lf_labels != ABSTAIN])
            
            results[lf_name] = {
                'coverage': coverage,
                'accuracy': accuracy,
                'total_labels': np.sum(labeled_mask),
                'label_distribution': dict(label_counts),
                'abstain_rate': np.sum(lf_labels == ABSTAIN) / len(lf_labels)
            }
            
            print(f"   {lf_name}:")
            print(f"     Coverage: {coverage:.3f} ({np.sum(labeled_mask)}/{len(lf_labels)} samples)")
            print(f"     Accuracy: {accuracy:.3f}")
            print(f"     Labels: {dict(label_counts)}")
        
        return results

    def create_visualizations(self, results: Dict, L_train: np.ndarray):
        """Create visualizations for labeling function analysis"""
        print(" Creating labeling function visualizations...")
        
        os.makedirs("outputs", exist_ok=True)
        
        # 1. Coverage and Accuracy Comparison
        plt.figure(figsize=(15, 5))
        
        plt.subplot(1, 3, 1)
        lf_names = list(results.keys())
        coverages = [results[lf]['coverage'] for lf in lf_names]
        accuracies = [results[lf]['accuracy'] for lf in lf_names]
        
        x_pos = np.arange(len(lf_names))
        
        plt.bar(x_pos - 0.2, coverages, 0.4, label='Coverage', alpha=0.7, color='skyblue')
        plt.bar(x_pos + 0.2, accuracies, 0.4, label='Accuracy', alpha=0.7, color='lightcoral')
        
        plt.xlabel('Labeling Functions')
        plt.ylabel('Score')
        plt.title('Labeling Function Performance')
        plt.xticks(x_pos, [lf.replace('lf_', '').replace('_', ' ').title() for lf in lf_names], rotation=45)
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 2. Label Matrix Heatmap
        plt.subplot(1, 3, 2)
        # Show first 100 samples for visibility
        sample_size = min(100, L_train.shape[0])
        heatmap_data = L_train[:sample_size, :].T
        
        sns.heatmap(heatmap_data, 
                   yticklabels=[lf.replace('lf_', '').replace('_', ' ').title() for lf in lf_names],
                   cmap='RdYlBu', center=0, 
                   cbar_kws={'label': 'Label Value'})
        plt.title(f'Label Matrix (First {sample_size} Samples)')
        plt.xlabel('Sample Index')
        
        # 3. Labeling Function Agreement
        plt.subplot(1, 3, 3)
        # Count how many LFs agree on each sample
        non_abstain_counts = np.sum(L_train != ABSTAIN, axis=1)
        
        plt.hist(non_abstain_counts, bins=range(len(lf_names)+2), alpha=0.7, color='steelblue')
        plt.xlabel('Number of Labeling Functions Applied')
        plt.ylabel('Number of Samples')
        plt.title('Labeling Function Agreement')
        plt.xticks(range(len(lf_names)+1))
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('outputs/question2_labeling_functions.png', dpi=300, bbox_inches='tight')
        plt.show()

    def log_to_wandb(self, results: Dict, L_train: np.ndarray):
        """Log labeling function results to W&B"""
        print(" Logging labeling function results to W&B...")
        
        # Log summary metrics for each LF
        for lf_name, metrics in results.items():
            wandb.log({
                f"{lf_name}_coverage": metrics['coverage'],
                f"{lf_name}_accuracy": metrics['accuracy'], 
                f"{lf_name}_total_labels": metrics['total_labels'],
                f"{lf_name}_abstain_rate": metrics['abstain_rate']
            })
        
        # Log overall statistics
        overall_coverage = np.sum(L_train != ABSTAIN) / L_train.size
        
        wandb.log({
            "overall_coverage": overall_coverage,
            "total_samples": L_train.shape[0],
            "num_labeling_functions": L_train.shape[1],
            "label_matrix_density": 1 - (np.sum(L_train == ABSTAIN) / L_train.size)
        })
        
        # Log visualization
        wandb.log({
            "labeling_functions_analysis": wandb.Image("outputs/question2_labeling_functions.png")
        })
        
        # Create detailed table
        lf_table = wandb.Table(columns=["Labeling_Function", "Coverage", "Accuracy", "Total_Labels", "Abstain_Rate"])
        
        for lf_name, metrics in results.items():
            lf_table.add_data(
                lf_name.replace('lf_', '').replace('_', ' ').title(),
                f"{metrics['coverage']:.3f}",
                f"{metrics['accuracy']:.3f}",
                metrics['total_labels'],
                f"{metrics['abstain_rate']:.3f}"
            )
        
        wandb.log({"labeling_function_performance": lf_table})
        
        print(" All labeling function metrics logged to W&B!")

    def print_summary(self, results: Dict, L_train: np.ndarray):
        """Print detailed summary of labeling function analysis"""
        print("\n" + "="*60)
        print(" LABELING FUNCTIONS ANALYSIS SUMMARY")
        print("="*60)
        
        print(f"\n Implemented Labeling Functions:")
        for i, (lf_name, metrics) in enumerate(results.items(), 1):
            print(f"\n   {i}. {lf_name.replace('lf_', '').replace('_', ' ').title()}:")
            print(f"      - Coverage: {metrics['coverage']:.1%} ({metrics['total_labels']} samples)")
            print(f"      - Accuracy: {metrics['accuracy']:.1%}")
            print(f"      - Labels: {metrics['label_distribution']}")
        
        print(f"\n Overall Statistics:")
        total_labels = np.sum(L_train != ABSTAIN) 
        total_possible = L_train.size
        print(f"   - Total label applications: {total_labels:,} / {total_possible:,}")
        print(f"   - Overall coverage: {total_labels/total_possible:.1%}")
        print(f"   - Matrix shape: {L_train.shape}")
        
        print(f"\n Question 2 completed! Labeling functions ready for aggregation.")


# Define labeling functions outside the class (required by Snorkel)
@labeling_function()
def lf_year_detector(x):
    """
    Labeling Function 1: Year Detection (1900-2099)
    
    WHAT: Detects 4-digit years in the specified range
    WHY: Years are often MISC entities in NER (e.g., "Olympics 2024", "War of 1812")
    HOW: Use regex to find 4-digit numbers in range 1900-2099
    
    Returns:
    - MISC if year found in sentence
    - ABSTAIN if no year found (let other functions decide)
    """
    sentence = x.sentence
    
    # Regex pattern for years 1900-2099
    # \b ensures word boundaries (not part of larger number)
    year_pattern = r'\b(19[0-9]{2}|20[0-9]{2})\b'
    
    if re.search(year_pattern, sentence):
        return MISC
    
    return ABSTAIN

@labeling_function()  
def lf_organization_detector(x):
    """
    Labeling Function 2: Organization Suffix Detection
    
    WHAT: Identifies organizations by common corporate suffixes
    WHY: Organizations often have legal suffixes that are strong indicators
    HOW: Pattern matching for Inc., Corp., Ltd., LLC, etc.
    
    Returns:
    - NOT_ENTITY for clear organization indicators (they should be ORG, not MISC)
    - ABSTAIN if no organization pattern found
    """
    sentence = x.sentence
    
    # Common organization suffixes (case-insensitive)
    org_suffixes = [
        r'\bInc\.?\b',      # Inc or Inc.
        r'\bCorp\.?\b',     # Corp or Corp.  
        r'\bLtd\.?\b',      # Ltd or Ltd.
        r'\bLLC\b',         # LLC
        r'\bCo\.?\b',       # Co or Co.
        r'\bCompany\b',     # Company
        r'\bCorporation\b', # Corporation
        r'\bLimited\b',     # Limited
        r'\bFoundation\b',  # Foundation
        r'\bInstitute\b'    # Institute
    ]
    
    # Check for any organization suffix
    for suffix_pattern in org_suffixes:
        if re.search(suffix_pattern, sentence, re.IGNORECASE):
            return NOT_ENTITY  # Organizations are NOT MISC entities
    
    return ABSTAIN

@labeling_function()
def lf_misc_keywords(x):
    """
    Additional Labeling Function: MISC Entity Keywords
    
    WHAT: Identifies common MISC entities by keywords
    WHY: MISC includes events, nationalities, languages, etc.
    HOW: Keyword matching for common MISC entity types
    """
    sentence = x.sentence
    
    # Common MISC entity indicators
    misc_keywords = [
        # Languages
        r'\bEnglish\b', r'\bFrench\b', r'\bGerman\b', r'\bSpanish\b',
        # Nationalities  
        r'\bAmerican\b', r'\bBritish\b', r'\bGerman\b', r'\bFrench\b',
        # Events/Awards
        r'\bOlympics\b', r'\bWorld Cup\b', r'\bNobel\b', r'\bOscar\b',
        # Time periods
        r'\bChristmas\b', r'\bEaster\b', r'\bMonday\b', r'\bJanuary\b'
    ]
    
    for keyword in misc_keywords:
        if re.search(keyword, sentence, re.IGNORECASE):
            return MISC
    
    return ABSTAIN


# Add the missing methods to SnorkelLabelingFunctions class
def add_missing_methods():
    """Add missing methods to SnorkelLabelingFunctions class"""
    
    def evaluate_labeling_functions(self, df: pd.DataFrame, L_train: np.ndarray, lfs: List) -> Dict:
        """Evaluate coverage and accuracy of labeling functions"""
        print(" Evaluating labeling function performance...")
        
        results = {}
        
        for i, lf in enumerate(lfs):
            lf_name = lf.name
            lf_labels = L_train[:, i]
            
            # Calculate coverage (% of non-abstain labels)
            coverage = np.sum(lf_labels != ABSTAIN) / len(lf_labels)
            
            # Calculate accuracy on labeled examples
            # Compare with ground truth where LF didn't abstain
            labeled_mask = lf_labels != ABSTAIN
            
            if np.sum(labeled_mask) > 0:
                labeled_predictions = lf_labels[labeled_mask]
                labeled_ground_truth = df.loc[labeled_mask, 'ground_truth'].values
                
                accuracy = np.sum(labeled_predictions == labeled_ground_truth) / len(labeled_predictions)
            else:
                accuracy = 0.0
            
            # Label distribution
            label_counts = Counter(lf_labels[lf_labels != ABSTAIN])
            
            results[lf_name] = {
                'coverage': coverage,
                'accuracy': accuracy,
                'total_labels': np.sum(labeled_mask),
                'label_distribution': dict(label_counts),
                'abstain_rate': np.sum(lf_labels == ABSTAIN) / len(lf_labels)
            }
            
            print(f"   {lf_name}:")
            print(f"     Coverage: {coverage:.3f} ({np.sum(labeled_mask)}/{len(lf_labels)} samples)")
            print(f"     Accuracy: {accuracy:.3f}")
            print(f"     Labels: {dict(label_counts)}")
        
        return results

    def create_visualizations(self, results: Dict, L_train: np.ndarray):
        """Create visualizations for labeling function analysis"""
        print(" Creating labeling function visualizations...")
        
        os.makedirs("outputs", exist_ok=True)
        
        # 1. Coverage and Accuracy Comparison
        plt.figure(figsize=(15, 5))
        
        plt.subplot(1, 3, 1)
        lf_names = list(results.keys())
        coverages = [results[lf]['coverage'] for lf in lf_names]
        accuracies = [results[lf]['accuracy'] for lf in lf_names]
        
        x_pos = np.arange(len(lf_names))
        
        plt.bar(x_pos - 0.2, coverages, 0.4, label='Coverage', alpha=0.7, color='skyblue')
        plt.bar(x_pos + 0.2, accuracies, 0.4, label='Accuracy', alpha=0.7, color='lightcoral')
        
        plt.xlabel('Labeling Functions')
        plt.ylabel('Score')
        plt.title('Labeling Function Performance')
        plt.xticks(x_pos, [lf.replace('lf_', '').replace('_', ' ').title() for lf in lf_names], rotation=45)
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 2. Label Matrix Heatmap
        plt.subplot(1, 3, 2)
        # Show first 100 samples for visibility
        sample_size = min(100, L_train.shape[0])
        heatmap_data = L_train[:sample_size, :].T
        
        sns.heatmap(heatmap_data, 
                   yticklabels=[lf.replace('lf_', '').replace('_', ' ').title() for lf in lf_names],
                   cmap='RdYlBu', center=0, 
                   cbar_kws={'label': 'Label Value'})
        plt.title(f'Label Matrix (First {sample_size} Samples)')
        plt.xlabel('Sample Index')
        
        # 3. Labeling Function Agreement
        plt.subplot(1, 3, 3)
        # Count how many LFs agree on each sample
        non_abstain_counts = np.sum(L_train != ABSTAIN, axis=1)
        
        plt.hist(non_abstain_counts, bins=range(len(lf_names)+2), alpha=0.7, color='steelblue')
        plt.xlabel('Number of Labeling Functions Applied')
        plt.ylabel('Number of Samples')
        plt.title('Labeling Function Agreement')
        plt.xticks(range(len(lf_names)+1))
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('outputs/question2_labeling_functions.png', dpi=300, bbox_inches='tight')
        plt.show()

    def log_to_wandb(self, results: Dict, L_train: np.ndarray):
        """Log labeling function results to W&B"""
        print(" Logging labeling function results to W&B...")
        
        # Log summary metrics for each LF
        for lf_name, metrics in results.items():
            wandb.log({
                f"{lf_name}_coverage": metrics['coverage'],
                f"{lf_name}_accuracy": metrics['accuracy'], 
                f"{lf_name}_total_labels": metrics['total_labels'],
                f"{lf_name}_abstain_rate": metrics['abstain_rate']
            })
        
        # Log overall statistics
        overall_coverage = np.sum(L_train != ABSTAIN) / L_train.size
        
        wandb.log({
            "overall_coverage": overall_coverage,
            "total_samples": L_train.shape[0],
            "num_labeling_functions": L_train.shape[1],
            "label_matrix_density": 1 - (np.sum(L_train == ABSTAIN) / L_train.size)
        })
        
        # Log visualization
        wandb.log({
            "labeling_functions_analysis": wandb.Image("outputs/question2_labeling_functions.png")
        })
        
        # Create detailed table
        lf_table = wandb.Table(columns=["Labeling_Function", "Coverage", "Accuracy", "Total_Labels", "Abstain_Rate"])
        
        for lf_name, metrics in results.items():
            lf_table.add_data(
                lf_name.replace('lf_', '').replace('_', ' ').title(),
                f"{metrics['coverage']:.3f}",
                f"{metrics['accuracy']:.3f}",
                metrics['total_labels'],
                f"{metrics['abstain_rate']:.3f}"
            )
        
        wandb.log({"labeling_function_performance": lf_table})
        
        print(" All labeling function metrics logged to W&B!")

    def print_summary(self, results: Dict, L_train: np.ndarray):
        """Print detailed summary of labeling function analysis"""
        print("\n" + "="*60)
        print(" LABELING FUNCTIONS ANALYSIS SUMMARY")
        print("="*60)
        
        print(f"\n Implemented Labeling Functions:")
        for i, (lf_name, metrics) in enumerate(results.items(), 1):
            print(f"\n   {i}. {lf_name.replace('lf_', '').replace('_', ' ').title()}:")
            print(f"      - Coverage: {metrics['coverage']:.1%} ({metrics['total_labels']} samples)")
            print(f"      - Accuracy: {metrics['accuracy']:.1%}")
            print(f"      - Labels: {metrics['label_distribution']}")
        
        print(f"\n Overall Statistics:")
        total_labels = np.sum(L_train != ABSTAIN) 
        total_possible = L_train.size
        print(f"   - Total label applications: {total_labels:,} / {total_possible:,}")
        print(f"   - Overall coverage: {total_labels/total_possible:.1%}")
        print(f"   - Matrix shape: {L_train.shape}")
        
        print(f"\n Question 2 completed! Labeling functions ready for aggregation.")

def main():
    """Main function for Question 2: Snorkel Labeling Functions"""
    print(" Starting Question 2: Snorkel AI Labeling Functions")
    print("="*60)
    
    # Initialize labeling functions class
    snorkel_lf = SnorkelLabelingFunctions()
    
    try:
        # Step 1: Initialize W&B
        snorkel_lf.initialize_wandb()
        
        # Step 2: Load and prepare data
        df = snorkel_lf.load_conll_data()
        
        # Step 3: Apply labeling functions
        L_train, lfs = snorkel_lf.apply_labeling_functions(df)
        
        # Step 4: Evaluate performance
        results = snorkel_lf.evaluate_labeling_functions(df, L_train, lfs)
        
        # Step 5: Create visualizations  
        snorkel_lf.create_visualizations(results, L_train)
        
        # Step 6: Log to W&B
        snorkel_lf.log_to_wandb(results, L_train)
        
        # Step 7: Print summary
        snorkel_lf.print_summary(results, L_train)
        
        # Save label matrix for Question 3
        np.save('outputs/label_matrix_Q2.npy', L_train)
        df.to_pickle('outputs/dataframe_Q2.pkl')
        
        print(f"\n Question 2 completed successfully!")
        print(f" Label matrix saved to outputs/ for Question 3")
        print(f" View W&B dashboard: https://wandb.ai/{wandb.run.entity}/{wandb.run.project}")
        
    except Exception as e:
        print(f" Error in Question 2: {e}")
        raise
    
    finally:
        # Finish W&B run
        wandb.finish()

if __name__ == "__main__":
    main()
