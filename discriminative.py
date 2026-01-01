from typing import List, Dict, Any, Literal, Union
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
                "projected": X_proj,
                "coeffs": lda.coef_,
                "coeff_norm": np.linalg.norm(lda.coef_),
                "explained_variance": lda.explained_variance_ratio_,
                "accuracy": accuracy_score(self.y, y_pred),
                "confusion_matrix": confusion_matrix(self.y, y_pred),
                "classification_report": classification_report(self.y, y_pred, output_dict=True),
            }
    
        except Exception as e:
            print(f"Layer {layer} failed: {e}")
            return {
                "layer": layer,
                "projected": "error",
                "coeffs": "error",
                "coeff_norm": 0.0,
                "explained_variance": "error",
                "accuracy": 0.0,
                "confusion_matrix": "error",
                "classification_report": "error",
            }

    def _sweep_linear_discriminative_projection(self,
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
        Computes an averaged steering vector from contrastive activations.

        """

        if normalize:
            steer_vec = (steering_coeffs * steering_vec) / np.linalg.norm(steering_vec)
        else:
            steer_vec = steering_coeffs * steer_vec

        return steer_vec
    
    def hook_model_with_discriminative_steer(self, steering_coeffs: float, normalize: bool = False, explicit_layers=None): 
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
        
    def decode_model_responses(prompts): 
        
        """
        Takes in a list of prompts and performs a forward pass of the model responses, save the resulting answers. 
        """
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
        
        self._compute_projected_vector()
        
    
    def _compute_projected_vector(self): 
        
        steer_vector = self.steerer._retrieve_steering_vector(explicit_layers=self.layers_to_visualize)
        self.proj_vector = self.steer.X @ steer_vector 
            # X is m rows, n features @ n_features x 1 = (m_rows x 1)
        return self.proj_vector 
    

    def plot_linear_discriminants(projected_data: np.ndarray, 
                                labels: np.ndarray,
                                plot_title: str,
                                label_dict: Optional[Dict[int, str]] = None,
                                alpha: float = 0.85):
        """
        Plot LDA-projected data, color by original class labels.

        Args:
            projected_data: (N, D) array from lda.transform(X)
            labels: (N,) array of class labels (e.g. 0 / 1)
            plot_title: Title of the plot
            label_dict: Optional mapping {0: "negative", 1: "positive"}
            alpha: Scatter transparency
        """
        if label_dict is None:
            label_dict = {label: str(label) for label in np.unique(labels)}

        projected_data = np.asarray(projected_data)
        labels = np.asarray(labels)

        unique_labels = np.unique(labels)

        if projected_data.shape[1] == 1:
            x = projected_data[:, 0]

            for label in unique_labels:
                mask = labels == label
                plt.scatter(x[mask], np.zeros(mask.sum()), alpha=alpha, label=label_dict.get(label, str(label)))

                class_mean = x[mask].mean()
                plt.scatter(class_mean, 0, color="black", zorder=5)
                plt.text(class_mean, 0.02, f"μ = {class_mean:.2f}", ha="center")

            if len(unique_labels) == 2:
                mean_0 = x[labels == unique_labels[0]].mean()
                mean_1 = x[labels == unique_labels[1]].mean()
                decision_boundary = 0.5 * (mean_0 + mean_1)
                plt.axvline(decision_boundary, linestyle="--", color="black", label="Decision midpoint")

            plt.yticks([])
            plt.xlabel("Linear Discriminant 1")


        else:
            for label in unique_labels:
                mask = labels == label
                plt.scatter(projected_data[mask, 0], projected_data[mask, 1], alpha=alpha, label=label_dict.get(label, str(label)), edgecolors="none")

                centroid = projected_data[mask, :2].mean(axis=0)
                plt.scatter(centroid[0],centroid[1],color="black",zorder=5)

            plt.xlabel("Linear Discriminant 1")
            plt.ylabel("Linear Discriminant 2")

        plt.title(plot_title)
        plt.legend()
        plt.ylim(-0.4, len(unique_labels) - 1 + 0.4)
        plt.grid(False)
        plt.tight_layout()
        plt.show()
            
            
        