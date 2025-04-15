import os
import pandas as pd
import bz2
import random
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns

class AmazonReviewsExplorer:
    def __init__(self, data_dir="./data"):
        self.data_dir = data_dir
        os.makedirs(data_dir, exist_ok=True)
        
        # Kaggle dataset paths
        self.kaggle_dir = "/Users/tranduythanh/.cache/kagglehub/datasets/bittlingmayer/amazonreviews/versions/7"
        self.train_file = os.path.join(self.kaggle_dir, "train.ft.txt.bz2")
        self.test_file = os.path.join(self.kaggle_dir, "test.ft.txt.bz2")
    
    def read_dataset(self, num_samples=50000):
        """Read samples from both train and test files"""
        print("Reading dataset samples...")
        
        train_samples = self._read_bz2_sample(self.train_file, num_samples)
        test_samples = self._read_bz2_sample(self.test_file, num_samples // 5)  # Smaller test set
        
        train_df = self._parse_fasttext_format(train_samples)
        test_df = self._parse_fasttext_format(test_samples)
        
        return train_df, test_df
    
    def explore_dataset(self, df, dataset_type=""):
        """Explore and print dataset statistics"""
        if dataset_type:
            print(f"\n=== {dataset_type} Dataset Analysis ===")
        
        print(f"\nTotal reviews: {len(df)}")
        print(f"Columns: {df.columns.tolist()}")
        
        # Sample reviews with more details
        print("\nSample reviews (first 10):")
        print("\n{:<10} {:<10} {:<100}".format("Index", "Label", "Text Preview"))
        print("-" * 120)
        for i, row in df.head(10).iterrows():
            text_preview = row['text'][:100] + "..." if len(row['text']) > 100 else row['text']
            print("{:<10} {:<10} {:<100}".format(i, row['label'], text_preview))
        
        # Label distribution
        print("\nLabel distribution:")
        label_counts = df['label'].value_counts()
        print(label_counts)
        
        # Text length analysis
        df['text_length'] = df['text'].apply(len)
        print("\nText length statistics:")
        print(df['text_length'].describe())
        
        # Word frequency analysis
        print("\nWord frequency analysis:")
        all_words = ' '.join(df['text']).lower().split()
        word_freq = Counter(all_words)
        print("\nTop 20 most common words:")
        for word, count in word_freq.most_common(20):
            print(f"{word}: {count}")
        
        # Generate visualizations
        self._generate_visualizations(df, dataset_type)
        
        # Save sample data to CSV for reference
        sample_file = os.path.join(self.data_dir, f"{dataset_type.lower().replace(' ', '_')}_samples.csv")
        df.head(10).to_csv(sample_file, index=False)
        print(f"\nSample data saved to: {sample_file}")
    
    def _read_bz2_sample(self, file_path, num_samples, seed=42):
        """Read a sample of lines from a bz2 file"""
        random.seed(seed)
        
        with bz2.open(file_path, 'rt', encoding='utf-8') as f:
            lines = f.readlines()
        
        return random.sample(lines, min(num_samples, len(lines)))
    
    def _parse_fasttext_format(self, lines):
        """Parse FastText format lines into a DataFrame"""
        data = []
        for line in lines:
            line = line.strip()
            if line:
                parts = line.split(' ', 1)
                if len(parts) == 2:
                    label = parts[0].replace('__label__', '')
                    text = parts[1]
                    data.append({'label': label, 'text': text})
        
        return pd.DataFrame(data)
    
    def _generate_visualizations(self, df, dataset_type):
        """Generate visualizations for the dataset"""
        # Create visualization directory
        viz_dir = os.path.join(self.data_dir, "visualizations")
        os.makedirs(viz_dir, exist_ok=True)
        
        # Label distribution plot
        plt.figure(figsize=(10, 6))
        sns.countplot(x='label', data=df)
        plt.title(f'Label Distribution - {dataset_type}')
        plt.savefig(os.path.join(viz_dir, f'{dataset_type.lower().replace(" ", "_")}_label_dist.png'))
        plt.close()
        
        # Text length distribution plot
        plt.figure(figsize=(10, 6))
        sns.histplot(df['text_length'], bins=50)
        plt.title(f'Text Length Distribution - {dataset_type}')
        plt.xlabel('Text Length')
        plt.ylabel('Count')
        plt.savefig(os.path.join(viz_dir, f'{dataset_type.lower().replace(" ", "_")}_text_length.png'))
        plt.close()

def main():
    print("Amazon Reviews Dataset Explorer")
    explorer = AmazonReviewsExplorer()
    
    # Read and explore the dataset
    train_df, test_df = explorer.read_dataset()
    
    # Explore training set
    explorer.explore_dataset(train_df, "Training Set")
    
    # Explore test set
    explorer.explore_dataset(test_df, "Test Set")
    
    print("\nExploration complete! Visualizations and sample data have been saved to the 'data' directory.")

if __name__ == "__main__":
    main()
