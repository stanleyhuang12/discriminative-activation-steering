from typing import List, Dict, Any, Literal, Union, Optional
import numpy as np
import pandas as pd
import torch
import os
import time 
import pickle 

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

from transformer_lens import HookedTransformer, ActivationCache
from transformer_lens.hook_points import HookPoint

import matplotlib.pyplot as plt 
import seaborn as sns
from math import ceil 
import warnings
warnings.filterwarnings("ignore", category=UserWarning)



class ContrastiveSteerer: 
    def __init__(self, model_name: str, model: Optional[HookedTransformer] = None ): 
        self.model_name = model_name 
        if model:
            self.model = model
        else: 
            HookedTransformer.from_pretrained(model_name)
        
        self._set_reproducibility()
    
    def _set_reproducibility(self, seed:int = 821): 
        torch.manual_seed(seed)
        np.random.seed(seed)
        
    def _compute_contrastive_steering_vector(self, pos, neg): 
        return (pos - neg)
    
    def _average_contrastive_steering_vector(self, vec): 
        return np.mean(vec, axis=1)
    
    
        

class DiscriminativeSteerer:
    """
    End-to-end class for:
      1) Extracting contrastive or different classes' residual streams
      2) Find the Fisher-maximizing critierion vector as the steering vector 
      4) Injecting steering vectors via permanent hooks
    """

    def __init__(self, model_name: str):
        self.model_name = model_name
        self.model = HookedTransformer.from_pretrained(model_name=model_name)
        
        self._perma_hook_initialized = False 
        self._set_reproducibility()
    
    def _set_reproducibility(self, seed:int = 821): 
        torch.manual_seed(seed)
        np.random.seed(seed)
    
    def extract_activations_from_prompts(self, df: pd.DataFrame, n_pairs: int): 
        df = df.copy()
    
        unique_pairs = df['pair_id'].unique()
        sampled_pairs = pd.Series(unique_pairs).sample(n=n_pairs, random_state=18).tolist()
        
        df_sample = df[df['pair_id'].isin(sampled_pairs)]
        df_sample = df_sample.sort_values('pair_id').reset_index(drop=True)
        
        self.prompts = df_sample['prompt'].to_list()
        
        self.logits, self.cache = self.model.run_with_cache(self.prompts)

        return self.logits, self.cache, self.model
        
    def retrieve_residual_stream_for_contrastive_pair(self, 
                                                      layer: Optional[int] = None,
                                                      positions_to_analyze: int = -1,
                                                      decompose_residual_stream: bool = False,
                                                      normalize_streams: bool = True):
        """
        Returns:
            pos, neg: Arrays of shape [n_layers, n_pairs, d_model]
        """
        batch_size = self.cache["hook_embed"].size(0)
        assert batch_size % 2 == 0, "Contrastive pairs must have even batch size."

        n_pairs = batch_size // 2

        if decompose_residual_stream:
            print("Decompose residual error not yet supported")
            accum_resids = self.cache.decompose_resid(apply_ln=normalize_streams) 
        else:
            accum_resids = self.cache.accumulated_resid(apply_ln=normalize_streams,)

        if layer is not None:
            resids = accum_resids[layer:layer + 1]
        else:
            resids = accum_resids

        # Result shape: [n_layers, batch, d_model]
        resids = resids[:, :, positions_to_analyze, :]

        # Reshape batch into pairs: [n_layers, n_pairs, 2, d_model]
        n_layers = resids.size(0)
        d_model = resids.size(-1)
        resids = resids.view(n_layers, n_pairs, 2, d_model)

        # By convention: index 1 = positive, 0 = negative
        pos = resids[:, :, 1, :]  # [n_layers, n_pairs, d_model]
        neg = resids[:, :, 0, :]  # [n_layers, n_pairs, d_model]

        return pos.detach().cpu().numpy(), neg.detach().cpu().numpy()
    
    def _compute_mean_difference(self, pos, neg): 
        """
        Internal method following Rimsky 2024 to take the mean difference of activations
        
        Args:
            pos: np.ndarray of shape (n_pairs, d_model)
            neg: np.ndarray of shape (n_pairs, d_model)

        Returns:
            mean_diff: np.ndarray of shape (1, d_model)
        """
        assert len(pos.shape == 2) & len(neg.shape == 2), "Mean difference takes a function of pos and neg activations in [n_pairs, d_models]"
        assert pos.shape == neg.shape, "pos and neg must have the same shape"
        
        pairwise_diff = pos - neg 
        
        print(pairwise_diff.shape)

        mean_diff = pairwise_diff.mean(axis=0)
        
        return mean_diff  # [n_layers, d_model]
        
        # [n_layers, n_pairs, d_model]

    def _linear_discriminative_projection(self, positions_to_analyze: int = -1, normalize_streams: bool = False,) -> Dict[str, Any]:
        """
        Internal method that runs linear discriminant analysis on contrastive residual streams for a single layer.
        Catches exceptions if LDA fails (e.g., degenerate inputs) and returns 'error'.
        """
        
        list_dict = []
        
        pos, neg = self.retrieve_residual_stream_for_contrastive_pair(
                positions_to_analyze=positions_to_analyze,
                decompose_residual_stream=False,
                normalize_streams=normalize_streams,
        )
        assert pos.shape[0] == neg.shape[0]
        
        n_layers = pos.shape[0]
        for l in range(n_layers): 
            pos_df = pd.DataFrame(pos[l])
            pos_df["is_syco"] = 1
            neg_df = pd.DataFrame(neg[l])
            neg_df["is_syco"] = 0

            mean_diff = self._compute_mean_difference(pos[l], neg[l])

            
            df = pd.concat([pos_df, neg_df], axis=0)
            self.y = df["is_syco"].values
            self.X = df.drop(columns=["is_syco"]).values
            try:
               
                lda = LinearDiscriminantAnalysis()
                
                X_proj = lda.fit_transform(self.X, self.y)
                y_pred = lda.predict(self.X)

                layer_dict = {
                    "layer": l,
                    "params": {
                        "projected": X_proj, 
                        "labels": self.y.copy(),
                        "coeffs": lda.coef_, 
                        "coeff_norm": np.linalg.norm(lda.coef_), 
                        "predictions": y_pred,
                        "scalings_": lda.scalings_
                    },
                    "caa": {
                        "mean_diff": mean_diff,
                        "mean_diff_norm": np.linalg.norm(mean_diff)
                    },
                    "explained_variance": lda.explained_variance_ratio_,
                    "accuracy": accuracy_score(self.y, y_pred),
                    "confusion_matrix": confusion_matrix(self.y, y_pred),
                    "classification_report": classification_report(self.y, y_pred, output_dict=True),
                }
                list_dict.append(layer_dict )
    
            except Exception as e:
                layer_dict = {
                    "layer": l,
                    "params": {
                        "projected": "error", 
                        "labels": "error",
                        "coeffs": "error", 
                        "coeff_norm": 0.0, 
                        "predictions": "error",
                        "scalings_": "error"
                    },
                    "caa": {
                        "mean_diff": mean_diff,
                        "mean_diff_norm": np.linalg.norm(mean_diff)
                    },
                    "coeffs": "error",
                    "coeff_norm": 0.0,
                    "explained_variance": "error",
                    "accuracy": 0.0,
                    "confusion_matrix": "error",
                    "classification_report": "error",
                    "error": e
                }
                list_dict.append(layer_dict)
            
        return list_dict

    def sweep_linear_discriminative_projection(self,
                                               save_dir: str,
                                               rule: Literal['layer', 'coeff_norm', 'explained_variance', 'accuracy'] = "accuracy", positions_to_analyze: int = -1, 
                                               normalize_streams: bool = False): 
        """
        Internal method that runs a sweep of the linear discriminant analysis across multiple layers, saves, and ranks results. 
        """
            
        self.cached_results = self._linear_discriminative_projection(positions_to_analyze=positions_to_analyze, normalize_streams=normalize_streams)
        self.cached_results.sort(key=lambda x: x[rule], reverse=True)
        
        self._save_cached_results(save_dir=save_dir)
        
        return self.cached_results
    
    def _initialize_perma_hook_model(self, layer: str, steering_vector: Union[np.ndarray, torch.Tensor]) -> HookedTransformer:
        """
        Registers a permanent hook that injects a steering vector to the HookedTransformer model.
        """
        if isinstance(steering_vector, np.ndarray):
            steering_vector = torch.tensor(steering_vector).detach()

        def steering_hook(activations, hook: HookPoint):
            return activations + steering_vector

        if not self.model: 
            self.model = HookedTransformer.from_pretrained(self.model_name)
        
        if self._perma_hook_initialized: 
            print("Perma hook already registered.")

        else: 
            self.model.add_perma_hook(name=layer, hook=steering_hook)
            self._perma_hook_initialized = True     

        return self.model
    
    def _save_cached_results(self, save_dir): 
        """
        Save cached LDA results to CSV. Handles layers that failed LDA ('error').
        """
        self.save_dir = save_dir
        os.makedirs(self.save_dir, exist_ok=True)
        experiment_id = int(time.time())
        
        self.file_path = f"{save_dir}/{self.model_name}_experiment_{experiment_id}.csv"
        with open(self.file_path, "wb") as f:
            pickle.dump(self.cached_results, f)

        print(f"Results saved to {self.file_path}")
        return self.cached_results
    
    
    def _retrieve_steering_vector(self, explicit_layers=None): 
        if self.cached_results: 
            if explicit_layers: 
                result_dict = [res for res in self.cached_results if res['layer'] == explicit_layers][0]
                coeffs = result_dict['params']['coeffs']
            else: 
                print("Retrieving the projection vector from the best accuracy. Only method supported for now.")
                self.cached_results.sort(key=lambda x: x["accuracy"], reverse=True)
                coeffs = self.cached_results[0]['coeffs']
        else: 
            file_path = os.path.join(self.save_dir, os.listdir(self.save_dir)[-1])
            with open(file_path, "rb") as f: 
                res = pickle.load(f)
                if explicit_layers: 
                    coeffs = res[explicit_layers]['params']['coeffs']
                else: 
                    coeffs = res[0]['params']['coeffs']
        return coeffs 
    
        
    @property 
    def _compute_discriminative_steering_vector(self, steering_vec: np.ndarray, steering_coeffs: float, normalize: bool = False) -> np.ndarray:
        """
        Computes a normalized discriminative steering vector. 

        """

        if normalize:
            steer_vec = (steering_coeffs * steering_vec) / np.linalg.norm(steering_vec)
        else:
            steer_vec = steering_coeffs * steer_vec

        return steer_vec
    
    def hook_discriminative_steer(self, steering_coeffs: float, normalize: bool = False, explicit_layers=None): 
        """
        Public method to call that permanently hooks the model with a steering vector. 
        """
        if steering_coeffs > 3: 
            print("Warning!: Steering coefficient may be too large, which can cause model responses to degenerate")
        steer_vector = self._retrieve_steering_vector(explicit_layers=explicit_layers)
        steer_vector = self._compute_discriminative_steering_vector(steering_vec=steer_vector, 
                                                                    steering_coeffs=steering_coeffs, 
                                                                    normalize=normalize)
        
        self._initialize_perma_hook_model(layer=explicit_layers, steering_vector=steer_vector)
        print(f"Registered a permanent forward hook to {self.model_name} model")
        
    def _compute_eigenvalue_from_cache(self, cache):
        projected = cache["params"]["projected"]
        labels = cache["params"]["labels"]

        if isinstance(projected, str) or (projected == 0).all():
            return None

        mu0 = projected[labels == 0].mean()
        mu1 = projected[labels == 1].mean()

        var0 = projected[labels == 0].var(ddof=1)
        var1 = projected[labels == 1].var(ddof=1)

        eigenvalue = (mu0 - mu1) ** 2 / (var0 + var1 + 1e-8)
        return float(eigenvalue)

    def _compute_layerwise_eigenvalues(self):
        layer_eigvals = []

        for cache in self.cached_results:
            eig = self._compute_eigenvalue_from_cache(cache)
            layer_eigvals.append(eig)

        return layer_eigvals
        
    def decode_model_responses(self, prompts, mode: Literal['inference', 'cache', 'hooks']): 
        
        """
        Takes in a list of prompts and performs a forward pass of the model responses, save the resulting answers. 
        """
        
        if self._perma_hook_initialized: print("Running models with registered permanent hooks.")
        
        if mode == "inference": 
            with torch.inference_mode():
                responses = self.model(prompts) 
                return responses 
        
        if mode == "cache": 
            self.model.run_with_cache(prompts)
            
        
        pass 


"""

We need the steering vector, activations of each layer, and then to multiply them together to get results 
We already computed it once and it is stored as a JSON dictionary object
"""

class DiscriminativeVisualizer: 
    """
    Module that takes a Discriminative Steerer and layers to visualize to create various plots for each layer 
    and visualizes how performant the discriminant axis is.
    """
    def __init__(self, steerer: DiscriminativeSteerer, layers_to_visualize: int):
        self.steerer = steerer
        self.layers_to_visualize = layers_to_visualize 
        
        self._deserialize_cached_results()
        
        
    def _deserialize_cached_results(self): 
        with open(self.steerer.file_path, "rb") as f: 
            self.results = pickle.load(f)
        return self.results 
    
    def plot_1d_layerwise_with_distribution(self,
                                            layer_to_projected: Dict[int, np.ndarray],
                                            labels: np.ndarray,
                                            plot_title: str,
                                            label_dict: Optional[Dict[int, str]] = None,
                                            alpha: float = 0.85,
                                            ):
        """
        Plot 1D LDA projections across layers as stacked subplots,
        with tiny distributions overlaid per class.

        Args:
            layer_to_projected: dict {layer_idx: (N, 1) or (N,)}
            labels: (N,)
            plot_title: overall figure title
            label_dict: optional {0: "neg", 1: "pos"}
            alpha: scatter transparency
        """

        layers = layer_to_projected.keys()
        n_layers = len(layers)
        n_cols = 2
        n_rows = ceil(n_layers / n_cols)
        classes = np.unique(labels)
        colors = sns.color_palette("tab10", n_colors=len(classes))

        fig, axes = plt.subplots(nrows=n_layers//2, 
                                 ncols=2,
                                 figsize=(8, 0.7 * n_layers),
                                 sharex=True,
                                 gridspec_kw={'hspace': 0.3, 'wspace': 0.2}  )
        
        axes = axes.flatten()

        for ax, layer in zip(axes, layers):
            x = layer_to_projected[layer]
            if not isinstance(x, (np.ndarray, list)) or (x == 0).all():
                print(f"Skipping layer {layer}: not numeric or invalid array")
                ax.set_yticks([])
                ax.set_ylim(0, 0.8)
                ax.spines["top"].set_visible(False)
                ax.spines["right"].set_visible(False)
                ax.spines["left"].set_visible(False)
                ax.set_ylabel(f"{layer}", rotation=0, labelpad=25, fontsize=9)
                ax.text(0.5, 0.5, "(not visualizable)", 
                        horizontalalignment='center', 
                        verticalalignment='center', 
                        transform=ax.transAxes,  
                        fontsize=8,
                        color='gray')
                continue

            x = x.squeeze()

            for i, cls in enumerate(classes):
                mask = labels == cls
                xi = x[mask]
                y = np.zeros(xi.shape)
                ax.scatter(xi,
                           y,
                           alpha=alpha,
                           s=12,
                           color=colors[i],
                           label=label_dict.get(cls, cls) if label_dict else cls)

                sns.kdeplot(x=x[mask], ax=ax,bw_adjust=0.75, clip=(x.min(), x.max()),fill=True, alpha=0.15, color=colors[i],linewidth=0)

            ax.set_yticks([])
            ax.set_ylim(0, 0.8)
            ax.set_ylabel(f"{layer}", rotation=0, labelpad=15, fontsize=7)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.spines["left"].set_visible(False)

        for j in range(len(layers), len(axes)):
            axes[j].axis("off")

        if label_dict:
            axes[0].legend(frameon=False,loc="upper right",fontsize=8)

        fig.suptitle(plot_title)
        fig.subplots_adjust(top=0.95)
        plt.tight_layout()
        plt.show()
    
    def plot_discriminative_projections(self, plot_title: str, label_dict: any, alpha:any=0.85): 
        self._deserialize_cached_results()


        sorted_result = sorted(self.results, key=lambda x: x['layer']) 
        layers_to_project = { }
        for res in sorted_result: 
            layer = res['layer']
            layer_name = f'layer_{layer}'
            projected = res['params']['projected']
            layers_to_project[layer_name] = projected
        
        labels = self.steerer.y
            
        self.plot_1d_layerwise_with_distribution(layer_to_projected=layers_to_project,
                                                 labels=labels,
                                                 plot_title=plot_title, 
                                                 label_dict=label_dict, 
                                                 alpha=alpha)
        
        return None 
    
    def plot_discriminability_per_layer(self, normalize_eigenvalues: bool = True):
        """
        Plot discriminability metrics (eigenvalues and accuracy) per layer.
        Supports normalization of eigenvalues.
        """

        cached = self.steerer.cached_results
        if not cached:
            raise ValueError("No cached results available to plot.")

        print(cached)
        eigvals = self.steerer._compute_layerwise_eigenvalues()
        cached = sorted(cached, key=lambda x: x['layer'])

        layers = []
        eigenvalues = []
        accuracy = []

        for cache, eig in zip(cached, eigvals):
            if eig is None or cache.get("accuracy") is None:
                continue
            layers.append(cache["layer"])
            eigenvalues.append(eig)
            accuracy.append(cache["accuracy"])

        if all(v is None for v in eigvals) and all(v is None for v in accuracy):
            raise ValueError("No eigenvalues or accuracy values found in cached results.")
        
        if normalize_eigenvalues:
            vals = np.array(eigenvalues, dtype=float)
            eigenvalues = (vals - vals.mean() / (vals.std() + 1e-8))

        eigenvalues = [np.nan if v is None else v for v in eigenvalues]
        accuracy = [np.nan if v is None else v for v in accuracy]

        # Plot
        plt.plot(
            layers,
            eigenvalues,
            marker="^",
            markersize=7,
            linewidth=2,
            label="Relative \nDiscriminability (Eigenvalue)" if not normalize_eigenvalues else "Normalized Discriminability (Eigenvalues)"
        )
        plt.plot(
            layers,
            accuracy,
            marker="o",
            markersize=6,
            linewidth=2,
            label="Accuracy"
        )

        plt.xlabel(f"Layers of {self.steerer.model_name}", fontsize=12)
        plt.ylabel("Value", fontsize=12)
        plt.legend(frameon=False)
        plt.grid(False)
        plt.tight_layout()
        plt.xticks(rotation=45, ha="right")
        plt.tick_params(axis="both", labelsize=10)
        plt.show()
            

    def plot_diagnostic_ablations(self): 
        """
        Creates subplots visualizations of head-wise loss of discriminative signals 
        under representational ablation. Note that for this method we are not guaranteeing
        causal ablation. 
        """
            
        pass 
    
    

class DiscriminatorEvaluator: 
    
    def __init__(self, steerer): 
        self.steerer = steerer
    
    @staticmethod 
    def _compute_pairwise_cosine_similarity(a, b):
        "Computes cosine similarity of two vectors"
    
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
    
    def compute_layerwise_cossim(self, metrics: Literal['caa', 'params', 'both']):
        """
        Computes cosine similarity across layers for steering vectors.

        Args:
            metrics: 'caa', 'params', or 'both'
                - 'caa': use contrastive activation addition vectors
                - 'params': use Fisher-derived vectors
                - 'both': plot both heatmaps side-by-side
        """

        cached_results = sorted(self.cached_results, key=lambda x: x['layer'], reverse=True)

        def compute_matrix(vector_key):
            all_vectors = [cache[vector_key] if vector_key == 'caa' else cache['params']['projected']
                        for cache in cached_results]
            all_vectors = np.array(all_vectors)
            n_layers = len(all_vectors)
            matrix = np.zeros((n_layers, n_layers))
            for i in range(n_layers):
                for j in range(n_layers):
                    matrix[i, j] = DiscriminatorEvaluator._compute_pairwise_cosine_similarity(
                        all_vectors[i], all_vectors[j]
                    )
            return matrix, n_layers

        if metrics in ['caa', 'params']:
            matrix, n_layers = compute_matrix(metrics)
            plt.figure(figsize=(4, 4))
            sns.heatmap(matrix, annot=False, cmap='coolwarm')
            plt.xticks(ticks=list(range(n_layers))[::5], labels=list(range(n_layers))[::5])
            plt.yticks(ticks=list(range(n_layers))[::5], labels=list(range(n_layers))[::5])
            plt.title(f"Layer similarity ({metrics}), {self.steerer.model_name}", fontsize=11)
            plt.show()

        elif metrics == 'both':
            fig, axes = plt.subplots(1, 2, figsize=(8, 4))

            for ax, m in zip(axes, ['caa', 'params']):
                matrix, n_layers = compute_matrix(m)
                sns.heatmap(matrix, annot=False, cmap='coolwarm', ax=ax)
                ax.set_xticks(list(range(n_layers))[::5])
                ax.set_yticks(list(range(n_layers))[::5])
                ax.set_title(f"{m} layer similarity")

            plt.suptitle(f"Layerwise Cosine Similarity, {self.steerer.model_name}", fontsize=12)
            plt.tight_layout()
            plt.show()

        else:
            raise ValueError("metrics must be one of 'caa', 'params', or 'both'")


    def compute_layerwise_comparison(self, metric: Literal['cosine', 'l2'] = 'cosine'):
        """
        Compare CAA vectors vs Fisher-derived vectors layer by layer.

        Args:
            metric: 'cosine' or 'l2'
                - 'cosine': cosine similarity between vectors
                - 'l2': Euclidean distance between vectors

        Returns:
            layer_metrics: np.ndarray of shape (n_layers,)
        """

        cached_results = sorted(self.cached_results, key=lambda x: x['layer'], reverse=True)
        n_layers = len(cached_results)
        layer_metrics = np.zeros(n_layers)

        for i, cache in enumerate(cached_results):
            caa_vec = cache['caa']['mean_diff'] 
            fisher_vec = cache['params']['projected']

            caa_vec = caa_vec.flatten()
            fisher_vec = fisher_vec.flatten()

            if metric == 'cosine':
                num = np.dot(caa_vec, fisher_vec)
                denom = np.linalg.norm(caa_vec) * np.linalg.norm(fisher_vec)
                layer_metrics[i] = num / denom
            elif metric == 'l2':
                layer_metrics[i] = np.linalg.norm(caa_vec - fisher_vec)
            else:
                raise ValueError("metric must be 'cosine' or 'l2'")

        plt.figure(figsize=(6, 3))
        plt.plot([cache['layer'] for cache in cached_results], layer_metrics, marker='o')
        plt.xlabel("Layer")
        plt.ylabel(f"{metric} between CAA and Fisher vectors")
        plt.title(f"Layerwise comparison: {self.steerer.model_name}")
        plt.grid(True)
        plt.show()

        return layer_metrics
