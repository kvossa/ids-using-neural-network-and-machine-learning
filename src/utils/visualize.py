import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix, roc_curve, auc, roc_auc_score
from sklearn.preprocessing import label_binarize
import plotly.graph_objects as go
from pathlib import Path

class IDSVisualizer:

	def __init__(self, output_dir='../../report/figures'):
		self.output_dir = Path(output_dir)
		self.output_dir.mkdir(parents=True, exist_ok=True)
		sns.set_style('whitegrid')
		plt.rcParams['figure.figsize'] = (12, 6)

	def save_fig(self, fig, filename):
		fig.savefig(self.output_dir / f"{filename}.png", dpi=300, bbox_inches='tight')
		fig.savefig(self.output_dir / f"{filename}.pdf")
		plt.close(fig)

	def plot_feature_distribution(self, df, feature, label_col='label'):
		fig, ax = plt.subplots()
		for label in df[label_col].unique():
			sns.kdeplot(df[df[label_col]==label][feature],
						label='Attack' if label else 'Normal', ax=ax)
			ax.set_title(f'Distribution of {feature}')
			ax.legend()
			self.save_fig(fig, f"feature_dist_{feature}")

	def plot_confusion_matrix(self, y_true, y_pred, classes=['Normal', 'Attack']):
		cm = confusion_matrix(y_true, y_pred)
		cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
		fig, ax = plt.subplots()
		sns.heatmap(cm_percent, annot=True, fmt='.1f', cmap='Blues',
					xticklabels=classes, yticklabels=classes)
		ax.set_xlabel('Predicted')
		ax.set_ylabel('Actual')
		ax.set_title('Confusion Matrix (%)')
		self.save_fig(fig, 'confusion_matrix')

	def plot_roc_curve(self, y_true, y_scores, classes):
		"""Interactive ROC curve with Plotly"""
		unique_classes = np.unique(y_true)
		y_true_bin = label_binarize(y_true, classes=unique_classes)		
		fig = go.Figure()

		auc_macro = roc_auc_score(y_true=y_true_bin, y_score=y_scores, average="macro", multi_class="ovr")
		auc_weighted = roc_auc_score(y_true=y_true_bin, y_score=y_scores, average="weighted", multi_class="ovr")

		for i in range(len(unique_classes)):
			fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_scores[:, i])
			roc_auc = auc(fpr, tpr)
			fig.add_trace(go.Scatter(x=fpr, y=tpr,
									mode='lines',
									name=f'{unique_classes[i]} (AUC = {roc_auc:.2f})'))

		fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1],
								mode='lines',
								line=dict(dash='dash'),
								name='Random'))
		fig.update_layout(
			title='Receiver Operating Characteristic',
			xaxis_title='False Positive Rate',
			yaxis_title='True Positive Rate',
			width=800,
			height=600
		)
		fig.write_html(str(self.output_dir / 'roc_curve.html'))
		return fig

	def plot_attack_timeline(self, df, time_col='timestamp', label_col='label'):
		"""Visualize attack patterns over time"""
		if time_col in df.columns:
			df[time_col] = pd.to_datetime(df[time_col])
			fig, ax = plt.subplots()
			df.set_index(time_col)[label_col].resample('1H').mean().plot(ax=ax)
			ax.set_title('Attack Ratio Over Time')
			ax.set_ylabel('Attack Probability')
			self.save_fig(fig, 'attack_timeline')

if __name__ == "__main__":
    viz = IDSVisualizer()
    
    import pandas as pd
    df = pd.DataFrame({
        'packet_size': np.concatenate([np.random.normal(100, 10, 1000),
                                      np.random.normal(200, 50, 200)]),
        'label': [0]*1000 + [1]*200
    })
    
    # Generate visualizations
    viz.plot_feature_distribution(df, 'packet_size')
    viz.plot_confusion_matrix([0,1,0,1], [0,0,1,1])
    viz.plot_roc_curve(df['label'], np.random.random(len(df)))

	