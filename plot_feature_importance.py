import joblib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def plot_feature_importance():
    model_path = 'models/trained_model.pkl'
    output_dir = 'plots'
    output_path = os.path.join(output_dir, 'manual_feature_importance.png')
    
    # Create plots directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}. Please run train_model.py first.")
        return

    # Load model payload
    print(f"Loading model from {model_path}...")
    payload = joblib.load(model_path)
    model = payload['model']
    features = payload['features']

    # Get feature importance
    importance_scores = model.feature_importances_
    
    # Create a DataFrame for plotting
    importance_df = pd.DataFrame({
        'Feature': features,
        'Importance': importance_scores
    })
    
    # Sort by importance
    importance_df = importance_df.sort_values(by='Importance', ascending=False)

    # Plotting
    plt.figure(figsize=(10, 8))
    sns.set_theme(style="whitegrid")
    
    # Use a premium color palette
    ax = sns.barplot(
        x='Importance', 
        y='Feature', 
        data=importance_df, 
        palette='viridis'
    )
    
    plt.title('AQI Model: Feature Importance Analysis', fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('Relative Importance Score', fontsize=12)
    plt.ylabel('Feature', fontsize=12)
    plt.tight_layout()

    # Save the plot
    plt.savefig(output_path, dpi=300)
    print(f"✅ Success! Feature importance plot saved to: {output_path}")
    
    # Print top features to console for quick reference
    print("\nTop 5 Most Important Features:")
    print(importance_df.head(5).to_string(index=False))

if __name__ == "__main__":
    plot_feature_importance()
