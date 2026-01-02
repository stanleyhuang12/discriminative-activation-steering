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



class DiscriminativeSteerer:
    """
    End-to-end class for:
      1) Extracting contrastive residual streams
      2) Computing difference / steering vectors
      3) Performing linear discriminative projections (LDA)
      4) Injecting steering vectors via permanent hooks
    """

    def __init__(self, model_name: str, d_model: int = 768):
        self.model_name = model_name
        self.d_model = d_model
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
                                                      layers: List[int], 
                                                      positions_to_analyze: int = -1,
                                                      decompose_residual_stream: bool = True,
                                                      normalize_streams: bool = True):
        """
        Returns:
            pos, neg: Arrays of shape [n_layers, n_pairs, d_model]
        """
        batch_size = self.cache["hook_embed"].size(0)
        assert batch_size % 2 == 0, "Contrastive pairs must have even batch size."

        n_pairs = batch_size // 2
        max_layer = max(layers)

        if decompose_residual_stream:
            accum_resids = self.cache.decompose_resid(layer=max_layer, apply_ln=normalize_streams)
        else:
            accum_resids = self.cache.accumulated_resid(layer=max_layer, apply_ln=normalize_streams,)

        # [n_layers, batch, pos, d_model]
        resids = accum_resids[-1] #TODO: right now only attn_out layer 
        resids = resids[:, positions_to_analyze, :]
        print(resids.size())
        resids = resids.view(n_pairs, 2, resids.size(-1))         # -> [n_layers, n_pairs, 2, d_model]
        pos = resids[:, 1, :]
        neg = resids[:, 0, :]
        
        return np.array(pos), np.array(neg)

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

    def _linear_discriminative_projection(self, layer: int, positions_to_analyze: int = -1, normalize_streams: bool = False,) -> Dict[str, Any]:
        """
        Internal method that runs linear discriminant analysis on contrastive residual streams for a single layer.
        Catches exceptions if LDA fails (e.g., degenerate inputs) and returns 'error'.
        """
        print(f"Running _linear_discriminative_projection for layer {layer}")
        
        try:
            pos, neg = self.retrieve_residual_stream_for_contrastive_pair(
                layers=[layer],
                positions_to_analyze=positions_to_analyze,
                decompose_residual_stream=True,
                normalize_streams=normalize_streams,
            )

            pos_df = pd.DataFrame(pos)
            pos_df["is_syco"] = 1
            print

            neg_df = pd.DataFrame(neg)
            neg_df["is_syco"] = 0

            df = pd.concat([pos_df, neg_df], axis=0)

            self.y = df["is_syco"].values
            self.X = df.drop(columns=["is_syco"]).values
            
            lda = LinearDiscriminantAnalysis()
            
            X_proj = lda.fit_transform(self.X, self.y)
            y_pred = lda.predict(self.X)

            return {
                "layer": layer,
                "params": {
                    "projected": X_proj, 
                    "coeffs": lda.coef_, 
                    "coeff_norm": np.linalg.norm(lda.coef_), 
                    "predictions": y_pred,
                    "scalings_": lda.scalings_
                },
                "explained_variance": lda.explained_variance_ratio_,
                "accuracy": accuracy_score(self.y, y_pred),
                "confusion_matrix": confusion_matrix(self.y, y_pred),
                "classification_report": classification_report(self.y, y_pred, output_dict=True),
            }
    
        except Exception as e:
            print(f"Layer {layer} failed: {e}")
            return {
                "layer": layer,
                "params": {
                    "projected": "error", 
                    "coeffs": "error", 
                    "coeff_norm": 0.0, 
                    "predictions": "error",
                    "scalings_": "error"
                },
                "coeffs": "error",
                "coeff_norm": 0.0,
                "explained_variance": "error",
                "accuracy": 0.0,
                "confusion_matrix": "error",
                "classification_report": "error",
            }

    def sweep_linear_discriminative_projection(self,
                                                save_dir: str,
                                                rule: Literal['layer', 'coeff_norm', 'explained_variance', 'accuracy'] = "accuracy", 
                                                positions_to_analyze: int = -1, 
                                                normalize_streams: bool = False): 
        """
        Internal method that runs a sweep of the linear discriminant analysis across multiple layers, saves, and ranks results. 
        """
            
        accumulated_residuals = self.cache.accumulated_resid()
        layers = accumulated_residuals.__len__()
        print(layers)
        self.cached_results = []
        
        for l in range(layers): 
            ret = self._linear_discriminative_projection(layer=l, positions_to_analyze=positions_to_analyze, normalize_streams=normalize_streams)
            self.cached_results.append(ret)
            
        self.cached_results.sort(key=lambda x: x[rule], reverse=True)
        
        self._save_cached_results(save_dir=save_dir)
        
        return self.cached_results
    
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
                coeffs = result_dict['coeffs']
            else: 
                print("Retrieving the projection vector from the best accuracy. Only method supported for now.")
                self.cached_results.sort(key=lambda x: x["accuracy"], reverse=True)
                coeffs = self.cached_results[0]['coeffs']
        else: 
            file_path = os.path.join(self.save_dir, os.listdir(self.save_dir)[-1])
            with open(file_path, "rb") as f: 
                res = pickle.load(f)
                if explicit_layers: 
                    coeffs = res[explicit_layers]['coeffs']
                else: 
                    coeffs = res[0]['coeffs']
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
        
    def _compute_eigenvalues(self, layers): 
        """
        Recover the raw eigenvalues from a fitted linear discriminant analysis. The eignevalue is computed as such: 
        
        S_b@W = \lambda * S_w@W 
        \lambda = S_b@W/S_w@W  
        
        The eigenvalue or lambda coefficient is also the ratio of projected vector's between-class variance and within-class variance. 
        While, it is hard to interpret the eigenvalue on its own, it could potentially offer a good metric for relative comparison 
         
        Relatively speaking, the higher the eigenvalue, the more robust the discriminative axis. 
        
        """
        
        projected = self.cached_results[layers]['params']['projected']
        label = self.y 
        
        assert projected.shape[1] == 1, "Only support two a maximum of K=2 classes"
        
        mu0 = projected[label == 0].mean()
        mu1 = projected[label == 1].mean()
        
        var0 = projected[y == 0].var(ddof=1)
        var1 = projected[y == 1].var(ddof=1)
       
        eigenvalue = (mu0 - mu1)**2 / (var0) + (var1)
        
        return eigenvalue 

    def _compute_layerwise_eigenvalues(self): 
        """
        Computes layerwise eigenvalues. 
        """
        
        layer_eigvals = []
        for cache in self.cached_results: 
            l = cache['layers']
            eigvals = self._compute_eigenvalues(l)
            layer_eigvals.append(eigvals)
            cache['eigenvalues'] = eigvals
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
                                            jitter: float = 0.02,
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
            jitter: vertical jitter for scatter
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
            if not isinstance(x, (np.ndarray, list)):
                print(f"Skipping layer {layer}: not numeric")
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
                y = np.random.normal(0, jitter, size=mask.sum())

                ax.scatter(x[mask],
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
    
    def plot_discriminative_projections(self, plot_title: str, label_dict: any, alpha:any=0.85, jitter:any=0.0): 
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
                                                 alpha=alpha,
                                                 jitter=jitter)
        
        return None 
    
    def plot_discriminability_per_layer(self, normalize: bool = True):
        """
        Plot discriminability metrics (eigenvalues and accuracy) per layer.
        """

        cached = self.steerer.cached_results
        if not cached:
            raise ValueError("No cached results available to plot.")

        eigvals = [l.get("eigenvalues") for l in cached]
        accuracy = [l.get("accuracy") for l in cached]
        layer_names = [l.get("layer") for l in cached]

        if all(v is None for v in eigvals) and all(v is None for v in accuracy):
            raise ValueError("No eigenvalues or accuracy values found in cached results.")


        def _normalize(vals):
            vals = np.array(vals, dtype=float)
            return (vals - vals.min()) / (vals.max() - vals.min() + 1e-8)

        if normalize:
            if any(v is not None for v in eigvals):
                eigvals = _normalize(eigvals)
            if any(v is not None for v in accuracy):
                accuracy = _normalize(accuracy)

        plt.figure(figsize=(9, 5))

        plt.plot(layer_names, eigvals, marker="^", markersize=7,linewidth=2, label="Discriminability (Eigenvalue)")
        plt.plot(layer_names, accuracy, marker="o", markersize=6, linewidth=2, label="Accuracy")

        plt.xlabel(f"Layers of {self.steerer.model_name}", fontsize=12)
        plt.ylabel("Normalized Value" if normalize else "Value", fontsize=12)

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
            
        