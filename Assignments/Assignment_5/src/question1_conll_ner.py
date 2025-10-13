"""
Question 1: CoNLL-2003 Named Entity Recognition Dataset Analysis
===============================================================

WHAT: This script loads the CoNLL-2003 NER dataset and analyzes entity distributions
WHY: CoNLL-2003 is a standard benchmark for NER tasks with 4 entity types
HOW: Use HuggingFace datasets to load data and W&B to track statistics

Entity Types:
- PER: Person names (e.g., "John Smith", "Barack Obama")
- LOC: Locations (e.g., "New York", "Paris")  
- ORG: Organizations (e.g., "Google", "United Nations")
- MISC: Miscellaneous entities (e.g., "Olympics", "Nobel Prize")

Steps:
1. Load CoNLL-2003 dataset from HuggingFace
2. Initialize W&B project "Q1-weak-supervision-ner"
3. Analyze dataset statistics (samples, entity distribution)
4. Log all metrics to W&B for tracking
"""

import wandb
import pandas as pd
import matplotlib.pyplot as plt
from datasets import load_dataset
from collections import Counter, defaultdict
import numpy as np
from typing import Dict, List, Tuple
import os

class CoNLLDatasetAnalyzer:
    """Analyzes CoNLL-2003 dataset and logs statistics to Weights & Biases"""
    
    def __init__(self):
        # Default entity labels mapping (BIO tagging scheme) - will be updated from dataset
        self.label_names = [
            'O',           # 0: Outside any entity
            'B-PER',       # 1: Beginning of person name
            'I-PER',       # 2: Inside person name
            'B-ORG',       # 3: Beginning of organization
            'I-ORG',       # 4: Inside organization  
            'B-LOC',       # 5: Beginning of location
            'I-LOC',       # 6: Inside location
            'B-MISC',      # 7: Beginning of miscellaneous entity
            'I-MISC'       # 8: Inside miscellaneous entity
        ]
        
        # Will be dynamically updated based on actual dataset labels
        self.entity_types = {}
        self.tokens_field = None  # Auto-detected field name for tokens
        self.labels_field = None  # Auto-detected field name for NER tags
        self.wandb_run = None     # Store wandb run reference safely

    def initialize_wandb(self):
        """Initialize Weights & Biases project for experiment tracking"""
        print(" Initializing Weights & Biases...")
        
        try:
            # Initialize W&B project with specific name as required
            self.wandb_run = wandb.init(
                project="Q1-weak-supervision-ner",  # Exact project name from assignment
                name="conll2003-dataset-analysis",
                tags=["dataset-analysis", "ner", "conll2003"],
                notes="Analysis of CoNLL-2003 dataset for Assignment 5 Question 1"
            )
            
            print(" W&B initialized successfully!")
            return self.wandb_run
            
        except Exception as e:
            print(f" W&B initialization failed: {e}")
            print(" Continuing without W&B logging...")
            self.wandb_run = None
            return None

    def load_dataset(self):
        """Load CoNLL-2003 dataset from HuggingFace with robust field detection"""
        print(" Loading CoNLL-2003 dataset from HuggingFace...")
        
        # List of dataset identifiers to try in order
        dataset_identifiers = [
            "conll2003",  # Standard identifier
            # "eriktks/conll2003",  # Original attempt
            # "nielsr/conll2003-ner",  # Alternative format
            # "tner/conll2003",  # Another alternative
        ]
        
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
                
                # Display basic dataset info
                print(f"Dataset splits: {list(dataset.keys())}")
                for split_name, split_data in dataset.items():
                    if len(split_data) > 0:
                        print(f"  - {split_name}: {len(split_data)} samples")
                
                # Auto-detect field names and validate structure
                if self._detect_dataset_structure(dataset):
                    return dataset
                else:
                    print(f" Dataset structure incompatible, trying next option...")
                    continue
                    
            except Exception as e:
                print(f" Failed to load '{dataset_id}': {e}")
                if i < len(dataset_identifiers) - 1:
                    print(" Trying next alternative...")
                    continue
                else:
                    print(" All dataset loading attempts failed!")
                    # As a last resort, provide instructions for manual setup
                    print("\n MANUAL SETUP INSTRUCTIONS:")
                    print("If all automatic loading fails, you can manually download the dataset:")
                    print("1. Visit: https://huggingface.co/datasets/conll2003")
                    print("2. Download the dataset files")
                    print("3. Place them in a 'data' directory")
                    print("4. Modify this script to load from local files")
                    raise Exception("Could not load CoNLL-2003 dataset from any source")
    
    def _detect_dataset_structure(self, dataset):
        """Auto-detect field names and extract label mappings"""
        try:
            # Find a non-empty split to examine
            sample = None
            for split_name, split_data in dataset.items():
                if len(split_data) > 0:
                    sample = split_data[0]
                    break
            
            if sample is None:
                print(" All dataset splits are empty!")
                return False
            
            # Detect tokens field
            possible_token_fields = ['tokens', 'words', 'text']
            self.tokens_field = None
            for field in possible_token_fields:
                if field in sample and isinstance(sample[field], (list, tuple)):
                    self.tokens_field = field
                    break
            
            # Detect labels field  
            possible_label_fields = ['ner_tags', 'tags', 'labels', 'ner_labels', 'entities']
            self.labels_field = None
            for field in possible_label_fields:
                if field in sample and isinstance(sample[field], (list, tuple)):
                    self.labels_field = field
                    break
            
            if not self.tokens_field or not self.labels_field:
                print(f"❌ Could not detect required fields. Available: {list(sample.keys())}")
                return False
            
            print(f" Detected tokens field: '{self.tokens_field}'")
            print(f" Detected labels field: '{self.labels_field}'")
            
            # Extract label names from dataset features (handles ClassLabel)
            self._extract_label_mappings(dataset)
            
            return True
            
        except Exception as e:
            print(f" Structure detection failed: {e}")
            return False
    
    def _extract_label_mappings(self, dataset):
        """Extract label names and create entity type mappings"""
        try:
            # Get the feature info for the labels field
            for split_name, split_data in dataset.items():
                if len(split_data) > 0:
                    features = split_data.features
                    break
            
            # Handle different label encoding formats
            if self.labels_field in features:
                label_feature = features[self.labels_field]
                
                # Handle Sequence(ClassLabel) format
                if hasattr(label_feature, 'feature') and hasattr(label_feature.feature, 'names'):
                    self.label_names = label_feature.feature.names
                # Handle ClassLabel format
                elif hasattr(label_feature, 'names'):
                    self.label_names = label_feature.names
                # Handle string labels or fallback to default
                else:
                    print("⚠️ Using default label names (could not extract from dataset)")
            
            print(f" Label names: {self.label_names}")
            
            # Create entity type mappings based on actual labels
            self.entity_types = {'O': []}
            
            for i, label_name in enumerate(self.label_names):
                if label_name == 'O':
                    self.entity_types['O'].append(i)
                elif '-' in label_name:
                    # Extract entity type (e.g., 'PER' from 'B-PER')
                    entity_type = label_name.split('-')[1]
                    if entity_type not in self.entity_types:
                        self.entity_types[entity_type] = []
                    self.entity_types[entity_type].append(i)
            
            print(f" Entity type mappings: {self.entity_types}")
            
        except Exception as e:
            print(f" Label mapping extraction failed: {e}")
            print(" Using default mappings...")
            # Fallback to default mappings
            self.entity_types = {
                'PER': [1, 2],    # Person: B-PER, I-PER
                'ORG': [3, 4],    # Organization: B-ORG, I-ORG
                'LOC': [5, 6],    # Location: B-LOC, I-LOC
                'MISC': [7, 8],   # Miscellaneous: B-MISC, I-MISC
                'O': [0]          # Outside: O
            }

    def analyze_dataset_statistics(self, dataset) -> Dict:
        """Analyze dataset statistics including samples and entity distribution"""
        print(" Analyzing dataset statistics...")
        
        stats = {
            'total_samples': 0,
            'split_sizes': {},
            'entity_distribution': defaultdict(int),
            'entity_counts_per_split': {},
            'token_statistics': {},
            'sentence_lengths': []
        }
        
        # Analyze each split (train, validation, test)
        for split_name, split_data in dataset.items():
            print(f"  Analyzing {split_name} split...")
            
            split_size = len(split_data)
            stats['total_samples'] += split_size
            stats['split_sizes'][split_name] = split_size
            
            # Skip empty splits
            if split_size == 0:
                print(f"    {split_name} split is empty, skipping...")
                continue
            
            # Count entities and tokens in this split
            split_entities = defaultdict(int)
            split_tokens = []
            split_sentences = []
            
            for sample in split_data:
                # Use auto-detected field names
                tokens = sample[self.tokens_field]
                labels = sample[self.labels_field]
                
                split_tokens.extend(tokens)
                split_sentences.append(len(tokens))
                
                # Count entity types using detected mappings
                for label in labels:
                    # Handle both integer and string labels
                    if isinstance(label, int) and label < len(self.label_names):
                        label_name = self.label_names[label]
                    else:
                        label_name = str(label)
                    
                    # Map to entity type
                    for entity_type, label_ids in self.entity_types.items():
                        if label in label_ids:
                            split_entities[entity_type] += 1
                            stats['entity_distribution'][entity_type] += 1
                            break
            
            stats['entity_counts_per_split'][split_name] = dict(split_entities)
            
            # Safe statistics calculation (handle empty cases)
            if split_sentences:
                stats['token_statistics'][split_name] = {
                    'total_tokens': len(split_tokens),
                    'unique_tokens': len(set(split_tokens)) if split_tokens else 0,
                    'avg_sentence_length': np.mean(split_sentences),
                    'max_sentence_length': max(split_sentences),
                    'min_sentence_length': min(split_sentences)
                }
                stats['sentence_lengths'].extend(split_sentences)
            else:
                stats['token_statistics'][split_name] = {
                    'total_tokens': 0,
                    'unique_tokens': 0,
                    'avg_sentence_length': 0.0,
                    'max_sentence_length': 0,
                    'min_sentence_length': 0
                }
        
        return stats

    def create_visualizations(self, stats: Dict):
        """Create visualizations for dataset analysis"""
        print(" Creating visualizations...")
        
        # Ensure outputs directory exists
        os.makedirs("outputs", exist_ok=True)
        
        # Get available entity types (exclude 'O' for the pie chart)
        available_entities = [et for et in self.entity_types.keys() if et != 'O' and stats['entity_distribution'][et] > 0]
        
        if not available_entities:
            print(" No entities found for visualization")
            return
        
        # 1. Entity Distribution Pie Chart
        plt.figure(figsize=(12, 5))
        
        plt.subplot(1, 2, 1)
        entity_counts = [stats['entity_distribution'][entity] for entity in available_entities]
        entity_labels = [f'{entity} ({entity})' for entity in available_entities]
        
        if sum(entity_counts) > 0:  # Avoid empty pie chart
            plt.pie(entity_counts, labels=entity_labels, autopct='%1.1f%%', startangle=90)
            plt.title('Entity Type Distribution in CoNLL-2003')
        else:
            plt.text(0.5, 0.5, 'No entities found', ha='center', va='center', transform=plt.gca().transAxes)
            plt.title('Entity Type Distribution (No Data)')
        
        # 2. Split Size Distribution
        plt.subplot(1, 2, 2)
        splits = list(stats['split_sizes'].keys())
        sizes = list(stats['split_sizes'].values())
        
        if splits and sizes:
            plt.bar(splits, sizes, color=['skyblue', 'lightcoral', 'lightgreen'][:len(splits)])
            plt.title('Dataset Split Sizes')
            plt.ylabel('Number of Samples')
            
            for i, size in enumerate(sizes):
                plt.text(i, size + max(sizes) * 0.01, str(size), ha='center', va='bottom')
        else:
            plt.text(0.5, 0.5, 'No splits found', ha='center', va='center', transform=plt.gca().transAxes)
            plt.title('Dataset Split Sizes (No Data)')
        
        plt.tight_layout()
        plt.savefig('outputs/question1_dataset_overview.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # 3. Sentence Length Distribution
        plt.figure(figsize=(10, 6))
        if stats['sentence_lengths']:  # Check if we have sentence length data
            plt.hist(stats['sentence_lengths'], bins=50, alpha=0.7, color='steelblue')
            mean_length = np.mean(stats['sentence_lengths'])
            plt.axvline(mean_length, color='red', linestyle='--', 
                       label=f'Mean: {mean_length:.1f}')
            plt.xlabel('Sentence Length (tokens)')
            plt.ylabel('Frequency')
            plt.title('Distribution of Sentence Lengths in CoNLL-2003')
            plt.legend()
            plt.grid(True, alpha=0.3)
        else:
            plt.text(0.5, 0.5, 'No sentence length data', ha='center', va='center', transform=plt.gca().transAxes)
            plt.title('Sentence Length Distribution (No Data)')
            
        plt.savefig('outputs/question1_sentence_lengths.png', dpi=300, bbox_inches='tight')
        plt.show()

    def log_to_wandb(self, stats: Dict):
        """Log all statistics to Weights & Biases"""
        print(" Logging statistics to W&B...")
        
        # Check if W&B is available
        if self.wandb_run is None:
            print(" W&B not available, skipping logging...")
            return
        
        # Safe calculation of total entities (avoid division by zero)
        total_entities = sum(stats['entity_distribution'].values())
        
        # Build summary metrics safely
        summary_metrics = {
            # Dataset size metrics
            "total_samples": stats['total_samples'],
            "train_samples": stats['split_sizes'].get('train', 0),
            "validation_samples": stats['split_sizes'].get('validation', 0),
            "test_samples": stats['split_sizes'].get('test', 0),
            
            # Entity distribution
            "total_entities": total_entities,
        }
        
        # Add entity counts for available types
        for entity_type in self.entity_types.keys():
            summary_metrics[f"{entity_type}_entities"] = stats['entity_distribution'][entity_type]
            
            # Add percentages only if we have entities
            if total_entities > 0:
                percentage = (stats['entity_distribution'][entity_type] / total_entities) * 100
                summary_metrics[f"{entity_type}_percentage"] = percentage
        
        # Add token statistics safely
        if stats['sentence_lengths']:
            summary_metrics.update({
                "avg_sentence_length": np.mean(stats['sentence_lengths']),
                "max_sentence_length": max(stats['sentence_lengths']),
                "min_sentence_length": min(stats['sentence_lengths']),
                "std_sentence_length": np.std(stats['sentence_lengths']),
            })
        
        # Log summary metrics
        wandb.summary.update(summary_metrics)
        
        # Log images if they exist
        import os
        if os.path.exists("outputs/question1_dataset_overview.png"):
            wandb.log({"dataset_overview": wandb.Image("outputs/question1_dataset_overview.png")})
        
        if os.path.exists("outputs/question1_sentence_lengths.png"):
            wandb.log({"sentence_length_distribution": wandb.Image("outputs/question1_sentence_lengths.png")})
        
        # Create and log a detailed table
        entity_table = wandb.Table(columns=["Entity_Type", "Count", "Percentage"])
        
        for entity_type in self.entity_types.keys():
            count = stats['entity_distribution'][entity_type]
            if total_entities > 0:
                percentage = (count / total_entities) * 100
                entity_table.add_data(entity_type, count, f"{percentage:.2f}%")
            else:
                entity_table.add_data(entity_type, count, "N/A")
        
        wandb.log({"entity_distribution_table": entity_table})
        
        print("✅ All statistics logged to W&B successfully!")

    def print_detailed_summary(self, stats: Dict):
        """Print detailed summary of dataset analysis"""
        print("\n" + "="*60)
        print(" DATASET ANALYSIS SUMMARY")
        print("="*60)
        
        print(f"\n Dataset Overview:")
        print(f"   Total Samples: {stats['total_samples']:,}")
        
        for split, size in stats['split_sizes'].items():
            if stats['total_samples'] > 0:
                percentage = (size / stats['total_samples']) * 100
                print(f"   {split.capitalize()}: {size:,} ({percentage:.1f}%)")
            else:
                print(f"   {split.capitalize()}: {size:,}")
        
        print(f"\n Entity Distribution:")
        total_entities = sum(stats['entity_distribution'].values())
        
        if total_entities > 0:
            for entity_type in self.entity_types.keys():
                if entity_type != 'O':  # Skip 'O' for main summary
                    count = stats['entity_distribution'][entity_type]
                    percentage = (count / total_entities) * 100
                    print(f"   {entity_type}: {count:,} ({percentage:.2f}%)")
        else:
            print("   No entities found in dataset")
        
        print(f"\n Token Statistics:")
        if stats['sentence_lengths']:
            avg_length = np.mean(stats['sentence_lengths'])
            print(f"   Average sentence length: {avg_length:.1f} tokens")
            print(f"   Shortest sentence: {min(stats['sentence_lengths'])} tokens")
            print(f"   Longest sentence: {max(stats['sentence_lengths'])} tokens")
        else:
            print("   No sentence length data available")
        
        print("\n✅ Analysis completed! Check W&B dashboard for detailed metrics.")

def main():
    """Main function to run the CoNLL-2003 dataset analysis"""
    print(" Starting Question 1: CoNLL-2003 NER Dataset Analysis")
    print("="*60)
    
    # Initialize analyzer
    analyzer = CoNLLDatasetAnalyzer()
    
    try:
        # Step 1: Initialize W&B
        analyzer.initialize_wandb()
        
        # Step 2: Load dataset
        dataset = analyzer.load_dataset()
        
        # Step 3: Analyze statistics
        stats = analyzer.analyze_dataset_statistics(dataset)
        
        # Step 4: Create visualizations
        analyzer.create_visualizations(stats)
        
        # Step 5: Log to W&B
        analyzer.log_to_wandb(stats)
        
        # Step 6: Print summary
        analyzer.print_detailed_summary(stats)
        
        print(f"\n Question 1 completed successfully!")
        print(f" View your W&B dashboard: https://wandb.ai/{wandb.run.entity}/{wandb.run.project}")
        
    except Exception as e:
        print(f" Error in Question 1: {e}")
        raise
    
    finally:
        # Finish W&B run
        wandb.finish()

if __name__ == "__main__":
    main()