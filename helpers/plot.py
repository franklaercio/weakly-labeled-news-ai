# =============================================================
# Author: Elias Jacob
# Some code adapted from Elias Jacob
# Repository: https://github.com/eliasjacob/imd3011-datacentric_ai
# License: MIT License
# =============================================================

import math

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.graph_objs as go
import plotly.subplots as sp
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import pyplot as plt
from plotly.subplots import make_subplots
from snorkel.labeling import LabelingFunction

# Constant representing abstaining from labeling
ABSTAIN = -1 

def plot_conflict_matrix(conflict_matrix_df):
    """
    Create an interactive visualization of the pairwise conflict matrix using Plotly,
    showing a smaller top-left section (approximately "one quarter" of dimensions).

    Args:
        conflict_matrix_df (pd.DataFrame): DataFrame containing the conflict matrix
        
    Returns:
        tuple: A tuple containing (heatmap_fig, summary_fig), the two Plotly figures generated.
    """
    # Determine sizes to extract a representative subset of the entire conflict matrix
    size1 = math.ceil(conflict_matrix_df.shape[0] / 2)
    size2 = math.floor(conflict_matrix_df.shape[0] / 2)

    # Extract a subsection (quarter) of the conflict matrix for visualization
    conflict_matrix_df_quarter = conflict_matrix_df.iloc[:size2, size2 :]

    # Create a heatmap using the extracted portion of the matrix
    heatmap_fig = go.Figure(data=go.Heatmap(
        z=conflict_matrix_df_quarter.values,
        x=conflict_matrix_df_quarter.columns,
        y=conflict_matrix_df_quarter.index,  
        colorscale='Reds',
        hovertemplate='LF1: %{y}<br>LF2: %{x}<br>Conflict Rate: %{z:.3f}<extra></extra>'
    ))
    # Update layout to include titles and adjust axis properties
    heatmap_fig.update_layout(
        title="Pairwise Conflicts Between Labeling Functions", 
        xaxis=dict(title="Labeling Functions", tickangle=45),
        yaxis=dict(title="Labeling Functions")
    )

    # Build a summary bar chart that aggregates total conflict scores per labeling function (LF)
    total_conflicts = []
    # Use the DataFrame index (LF names) to aggregate conflict counts per LF
    lf_names = conflict_matrix_df.index
    for lf in lf_names:
        # Sum conflicts across the row, subtracting the self-conflict entry
        conflict_sum = conflict_matrix_df.loc[lf].sum() - conflict_matrix_df.loc[lf, lf]
        total_conflicts.append(conflict_sum)

    # Create a summary DataFrame for easy plotting
    summary_df = pd.DataFrame({
        'Labeling Function': lf_names,
        'Total Conflicts': total_conflicts
    })

    # Sort the DataFrame in descending order of total conflicts
    summary_df = summary_df.sort_values('Total Conflicts', ascending=False)

    # Generate a bar chart from the summary DataFrame using Plotly Express
    summary_fig = px.bar(
        summary_df,
        x='Labeling Function',
        y='Total Conflicts',
        title='Total Conflicts by Labeling Function',
        color='Total Conflicts',
        color_continuous_scale='Reds'
    )
    # Update layout with x-axis rotation and y-axis title styling
    summary_fig.update_layout(
        xaxis=dict(title="Labeling Functions", tickangle=45),
        yaxis=dict(title="Total Conflict Score")
    )

    # Return both the heatmap and bar chart figures
    return heatmap_fig, summary_fig


def plot_coverage_overlap(labeling_functions: list[LabelingFunction], 
                          label_matrix: np.ndarray,
                          colorscale='Reds',
                          show_values=True,
                          sort_by_coverage=True) -> dict:
    """
    Visualize coverage and overlap between labeling functions.
    
    Args:
        labeling_functions (list[LabelingFunction]): A list of labeling functions.
        label_matrix (np.ndarray): A numpy array of shape (num_examples, num_lfs) with labels.
        colorscale (str or list): Colorscale for the heatmap (default: 'Reds').
        show_values (bool): Whether to show coverage values on the heatmap (default: True).
        sort_by_coverage (bool): Whether to sort LFs by their overall coverage (default: True).
        
    Returns:
        dict: A dictionary with overlap and coverage information, and the Plotly figure.
    """
    # Extract LF names from the list of labeling functions
    lf_names = [lf.name for lf in labeling_functions]
    n_lfs = len(labeling_functions)
    
    # Build a boolean mask to indicate coverage (non-ABSTAIN) per LF across all examples
    coverage_masks = [label_matrix[:, i] != ABSTAIN for i in range(n_lfs)]
    
    # Calculate the coverage rate (fraction of examples labeled) for each LF
    coverages = [mask.mean() for mask in coverage_masks]
    
    # Prepare indices for sorting, if needed
    indices = list(range(n_lfs))
    
    # If sorting by coverage is enabled, sort labeling functions in descending order based on their coverage
    if sort_by_coverage:
        sorted_indices = np.argsort(coverages)[::-1]  # Descending sort order
        indices = sorted_indices
    
    # Reorder LF names, masks, and coverage values using the sorted indices
    sorted_lf_names = [lf_names[i] for i in indices]
    sorted_masks = [coverage_masks[i] for i in indices]
    sorted_coverages = [coverages[i] for i in indices]
    
    # Initialize an overlap matrix to store pairwise overlap statistics between LFs
    overlap_matrix = np.zeros((n_lfs, n_lfs))
    
    # Calculate overlap rates: diagonal holds coverage rate, others hold pairwise overlap rates
    for i in range(n_lfs):
        for j in range(n_lfs):
            if i == j:
                # Diagonal: store individual LF coverage
                overlap_matrix[i, j] = sorted_coverages[i]
            else:
                # Compute overlap as the fraction of examples labeled by both LF i and LF j
                overlap_matrix[i, j] = (sorted_masks[i] & sorted_masks[j]).mean()
    
    # Convert the overlap matrix into a DataFrame for anyone needing tabular data
    overlap_df = pd.DataFrame(overlap_matrix, 
                              columns=sorted_lf_names, 
                              index=sorted_lf_names)
    
    # Create a DataFrame for LF-wise coverage for bar chart visualization later
    coverage_df = pd.DataFrame({
        'Labeling Function': sorted_lf_names,
        'Coverage': sorted_coverages
    })
    
    # Create a subplot layout with two columns: one for the heatmap, one for the coverage bar chart
    fig = sp.make_subplots(
        rows=1, cols=2,
        column_widths=[0.7, 0.3],
        subplot_titles=["Labeling Function Overlap", "Coverage by LF"],
        specs=[[{"type": "heatmap"}, {"type": "bar"}]]
    )
    
    # Prepare custom hover text for each cell in the heatmap with percentage formatting
    hover_text = []
    for i in range(n_lfs):
        row_texts = []
        for j in range(n_lfs):
            if i == j:
                # Diagonal cell: show individual coverage information
                row_texts.append(f"{sorted_lf_names[i]}<br>Coverage: {sorted_coverages[i]:.2%}")
            else:
                # Off-diagonal cell: show pairwise overlap info
                row_texts.append(f"Overlap: {overlap_matrix[i, j]:.2%}<br>{sorted_lf_names[i]} ∩ {sorted_lf_names[j]}")
        hover_text.append(row_texts)
    
    # Optionally create a text annotation matrix to display values on the heatmap
    text = None
    if show_values:
        text = []
        for i in range(n_lfs):
            row_text = []
            for j in range(n_lfs):
                row_text.append(f"{overlap_matrix[i, j]:.2f}")
            text.append(row_text)
    
    # Create a version of the overlap matrix for visualization and set the diagonal to None
    # so that the heatmap background remains transparent on the diagonal
    viz_matrix = overlap_matrix.copy()
    for i in range(n_lfs):
        viz_matrix[i, i] = None
    
    # Add the heatmap trace to the figure using the prepared matrices and hover texts
    heatmap = go.Heatmap(
        z=viz_matrix,
        x=sorted_lf_names,
        y=sorted_lf_names,
        colorscale=colorscale,
        text=text,
        hoverinfo='text',
        hovertext=hover_text,
        showscale=True,
        colorbar=dict(
            title='Overlap',
            thickness=15,
            tickformat='.0%'
        )
    )
    fig.add_trace(heatmap, row=1, col=1)
    
    # Add a bar chart trace for the LF coverage statistics
    bar = go.Bar(
        x=coverage_df['Coverage'],
        y=coverage_df['Labeling Function'],
        orientation='h',
        marker=dict(color='rgba(58, 71, 80, 0.8)'),
        text=[f"{v:.1%}" for v in coverage_df['Coverage']],
        textposition='auto',
        name='Coverage'
    )
    fig.add_trace(bar, row=1, col=2)
    
    # Update the overall layout for the subplots: titles, axis formatting, and legend placement
    fig.update_layout(
        title='Labeling Function Coverage and Overlap Analysis',
        xaxis=dict(tickangle=-45),
        xaxis2=dict(
            title='Coverage',
            tickformat='.0%',
            range=[0, max(coverages) * 1.1]  # Create a margin of 10% above max coverage
        )
    )
    
    # Reverse the y-axis order on the bar chart to match the order in the heatmap
    fig.update_yaxes(
        autorange="reversed",
        row=1, col=2
    )
    
    # Return a dictionary containing the overlap DataFrame, coverage DataFrame, and the generated figure
    return {
        'overlap_matrix': overlap_df,
        'coverage': coverage_df,
        'figure': fig
    }


def plot_confusion_matrix(matrix: np.ndarray) -> None:
    """
    Plot a confusion matrix using seaborn and matplotlib.

    Args:
        matrix (np.ndarray): Confusion matrix to plot.

    Returns:
        None
    """

    # Optional: class labels
    class_labels = ['World', 'Sport', 'Business', 'Sci-Tech']

    # Plot
    plt.figure(figsize=(6, 4))
    sns.heatmap(matrix, annot=True, fmt='d', cmap='Reds',
                xticklabels=class_labels, yticklabels=class_labels)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.show()


def analyze_model_predictions(predictions, true_labels, texts, model_name):
    """
    Analyzes model predictions against true labels, providing counts for
    correct, incorrect, and abstained predictions, and prints details
    for incorrect predictions.

    Args:
        predictions (np.array): Array of predicted labels from the model.
        true_labels (np.array): Array of ground truth labels.
        texts (np.array or pd.Series): Array or Series of text examples
                                       corresponding to the labels.
        model_name (str): Name of the model for display purposes.
    """
    if not isinstance(predictions, np.ndarray):
        predictions = np.array(predictions)
    if not isinstance(true_labels, np.ndarray):
        true_labels = np.array(true_labels)
    if not isinstance(texts, np.ndarray) and not isinstance(texts, pd.Series):
        texts = np.array(texts)
    elif isinstance(texts, pd.Series):
        texts = texts.values # Ensure it's an array for consistent indexing

    if len(predictions) != len(true_labels) or len(predictions) != len(texts):
        print("Error: Predictions, true_labels, and texts must have the same length.")
        return

    print(f"\n--- Analysis for {model_name} ---")

    # Calculate number of abstains
    num_abstains = np.count_nonzero(predictions == -1)
    print(f"Number of abstains by {model_name}: {num_abstains}")

    # Calculate correct predictions
    # A prediction is correct if prediction == true_label
    # This means if prediction is -1 and true_label is -1, it's "correct" by this definition.
    # If true_labels cannot be -1, then abstains will never be "correct" unless handled separately.
    correct_mask = (predictions == true_labels)
    num_correct = np.count_nonzero(correct_mask)
    print(f"Number of correct predictions by {model_name}: {num_correct} / {len(true_labels)}")

    # Calculate incorrect predictions
    # A prediction is incorrect if prediction != true_label
    incorrect_mask = (predictions != true_labels)
    num_incorrect = np.count_nonzero(incorrect_mask)
    idxs_incorrect = np.where(incorrect_mask)[0]
    print(f"Number of incorrect predictions by {model_name}: {num_incorrect} / {len(true_labels)}")

    if num_incorrect > 0:
        print(f"\nDetails of incorrect predictions by {model_name}:")
        # Limit the number of printed incorrect examples if there are too many
        max_errors_to_print = 10
        for i, idx in enumerate(idxs_incorrect):
            if i >= max_errors_to_print:
                print(f"... and {num_incorrect - max_errors_to_print} more incorrect predictions not shown.")
                break
            predicted_label = predictions[idx]
            true_label = true_labels[idx]
            text_example = texts[idx]

            print(f"\nIndex: {idx}")
            print(f"Predicted ({model_name}): {predicted_label}, True: {true_label}")
            print(f"Text: {text_example}")
            print("-" * 30)
    else:
        print(f"No incorrect predictions found for {model_name}!")    