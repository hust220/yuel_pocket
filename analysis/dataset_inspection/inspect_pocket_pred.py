# %%
import sys
import pickle
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
sys.path.append('../..')
from src.db_utils import db_connection

def analyze_pocket_predictions():
    """Analyze the distribution of pocket predictions in moad_test_results."""
    with db_connection() as conn:
        cur = conn.cursor()
        
        # Get all pocket predictions
        cur.execute("""
            SELECT pocket_pred 
            FROM moad_test_results
            WHERE pocket_pred IS NOT NULL
        """)
        results = cur.fetchall()
        
        # Process results
        max_probs = []
        all_probs = []
        neg_log_probs = []  # Store -log(p) values
        
        for (pocket_pred,) in results:
            pred_array = pickle.loads(pocket_pred)
            max_prob = np.max(pred_array)
            max_probs.append(max_prob)
            all_probs.extend(pred_array.squeeze().tolist())
            
            # Calculate -log(p) for max probability
            if max_prob > 0:  # Avoid log(0)
                neg_log_probs.append(-np.log(max_prob))
        
        max_probs = np.array(max_probs)
        all_probs = np.array(all_probs)
        neg_log_probs = np.array(neg_log_probs)
        
        # Create plots
        plt.figure(figsize=(15, 5))
        
        # Plot 1: Histogram of all probabilities
        plt.subplot(131)
        plt.hist(all_probs, bins=50, alpha=0.7)
        plt.xlabel('Probability')
        plt.ylabel('Frequency')
        plt.title('Distribution of All Probabilities')
        
        # Plot 2: Histogram of max probabilities
        plt.subplot(132)
        plt.hist(max_probs, bins=50, alpha=0.7)
        plt.xlabel('Maximum Probability')
        plt.ylabel('Frequency')
        plt.title('Distribution of Maximum Probabilities')
        
        # Plot 3: Histogram of -log(max_prob)
        plt.subplot(133)
        plt.hist(neg_log_probs, bins=50, alpha=0.7)
        plt.xlabel('-log(Maximum Probability)')
        plt.ylabel('Frequency')
        plt.title('Distribution of -log(Max Probabilities)')
        
        plt.tight_layout()
        plt.show()
        
        # Print statistics
        print("\nProbability Statistics:")
        print(f"All probabilities - min: {all_probs.min():.3e}, max: {all_probs.max():.3e}, mean: {all_probs.mean():.3e}")
        print(f"Max probabilities - min: {max_probs.min():.3e}, max: {max_probs.max():.3e}, mean: {max_probs.mean():.3e}")
        print(f"Number of predictions: {len(max_probs)}")
        
        # Calculate percentiles for max probabilities
        percentiles = [0, 1, 5, 10, 25, 50, 75, 90, 95, 99, 100]
        print("\nMax Probability Percentiles:")
        for p in percentiles:
            value = np.percentile(max_probs, p)
            print(f"{p}th percentile: {value:.3e} (-log: {-np.log(value) if value > 0 else 'inf'})")

if __name__ == "__main__":
    analyze_pocket_predictions()

# %%
