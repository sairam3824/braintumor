import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os

def draw_pipeline():
    fig, ax = plt.subplots(figsize=(8, 9))
    ax.axis('off')
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)

    # ----------------------------------------
    # Pipeline Steps
    # ----------------------------------------
    steps = [
        ("Database / Input", "1. MRI Images\n(4 Classes)"),
        ("Preprocessing", "2. Data Preprocessing\n(Grayscale, CLAHE 2.0, Resize 128x128)"),
        ("Deep Learning Feature Extractor", "3. Custom Autoencoder\n(Unsupervised CNN Bottleneck)"),
        ("Dimensionality Reduction", "4. PCA\n(Reduce Features to 50 Principal Components)"),
        ("Machine Learning Classifier", "5. RBF SVM Classifier\n(Support Vector Machine, C=300)"),
        ("Output", "6. Tumor Prediction\n(+ GradCAM Visual Explainability)")
    ]

    box_width = 7
    box_height = 0.8
    start_x = 1.5
    start_y = 8.5
    gap = 1.4
    
    for i, (layer_type, step) in enumerate(steps):
        y = start_y - (i * gap)
        
        # Draw arrow from previous box
        if i > 0:
            arrow_y_start = (start_y - ((i-1) * gap)) - box_height
            arrow_y_end = y
            
            plt.arrow(
                start_x + box_width/2, arrow_y_start, 
                0, arrow_y_end - arrow_y_start + 0.05, 
                head_width=0.3, head_length=0.25, 
                fc='black', ec='black', length_includes_head=True
            )

        # Highlight different parts of the pipeline with different colors
        if i == 0 or i == 5:
            fc_color = "#2d3748" # Dark Gray (I/O)
        elif i == 1:
            fc_color = "#3182ce" # Blue (Processing)
        elif i == 2:
            fc_color = "#38a169" # Green (Deep Learning)
        else:
            fc_color = "#d69e2e" # Yellow/Orange (Machine Learning)

        # Draw rounded box
        rect = patches.FancyBboxPatch(
            (start_x, y - box_height), box_width, box_height, 
            boxstyle="round,pad=0.2", ec="black", fc=fc_color, lw=2
        )
        ax.add_patch(rect)
        
        # Add descriptive text inside the box
        plt.text(
            start_x + box_width/2, y - box_height/2, step, 
            ha='center', va='center', fontsize=11, color='white', fontweight='bold'
        )

    plt.title("Brain Tumor Classification: Full Project Pipeline", fontsize=16, fontweight='bold')
    
    # Save the pipeline diagram
    diagrams_dir = os.path.join(os.path.dirname(__file__), "diagrams")
    os.makedirs(diagrams_dir, exist_ok=True)
    save_path = os.path.join(diagrams_dir, "project_pipeline.png")
    
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()
    
if __name__ == "__main__":
    draw_pipeline()
    print("Project Pipeline flow diagram saved successfully!")
