from typing import List, Dict, Any, Literal, Union
import numpy as np
import pandas as pd
import torch
import os
import time 
import csv
import json

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

from transformer_lens import HookedTransformer, ActivationCache
from transformer_lens.hook_points import HookPoint


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
        resids = accum_resids[:, positions_to_analyze, :]

        resids = resids.view(n_pairs, 2, resids.size(-1))         # -> [n_layers, n_pairs, 2, d_model]
        pos = resids[:, 1, :]
        neg = resids[:, 0, :]
        
        return np.array(pos)[layers, :, :], np.array(neg)[layers, :, :]

    @staticmethod
    def _compute_diff_vector(pos: np.ndarray, neg: np.ndarray) -> np.ndarray:
        """Compute difference of the vector, which forms the steering vector"""
        return pos - neg

    def _compute_diff_steering_vector(self, pos: np.ndarray, neg: np.ndarray, steering_coeffs: float, normalize: bool = False) -> np.ndarray:
        """
        Computes an averaged steering vector from contrastive activations.

        pos, neg:
            [n_layers, n_pairs, d_model]
        """
        diff = self._compute_diff_vector(pos, neg)

        average_diff = np.mean(diff[0], axis=0, keepdims=True)

        if normalize:
            steer_vec = (steering_coeffs * average_diff) / np.linalg.norm(average_diff)
        else:
            steer_vec = steering_coeffs * average_diff

        return steer_vec

    def initialize_perma_hook_model(self, layer: str, steering_vector: Union[np.ndarray, torch.Tensor]) -> HookedTransformer:
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

    def _linear_discriminative_projection(self, layer: int, positions_to_analyze: int = -1,  normalize_streams: bool = False,) -> Dict[str, Any]:
        """
        Internal method that runs linear discriminant analysis on contrastive residual streams for a single layer.
        """
        print("Running _linear_discriminative_projection")
        pos, neg = self.retrieve_residual_stream_for_contrastive_pair(
            layers=[layer],
            positions_to_analyze=positions_to_analyze,
            decompose_residual_stream=True,
            normalize_streams=normalize_streams,
        )

        pos_df = pd.DataFrame(pos[0])
        pos_df["is_syco"] = 1

        neg_df = pd.DataFrame(neg[0])
        neg_df["is_syco"] = 0

        df = pd.concat([pos_df, neg_df], axis=0)
        print(df)

        self.y = df["is_syco"].values
        self.X = df.drop(columns=["is_syco"]).values
        
        if self.X.shape[0] < self.X.shape[1]: 
            solver = "eigen"
        else: 
            solver = "svd"
            
        lda = LinearDiscriminantAnalysis(solver=solver)
        X_proj = lda.fit_transform(self.X, self.y)
        y_pred = lda.predict(self.X)
        print(y_pred)
        
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
            
        self.cached_results.sort(key=lambda x: x[rule])
        
        self._save_cached_results(save_dir=save_dir)
        
        return self.cached_results
    
    def _save_cached_results(self, save_dir): 
        
        os.makedirs(save_dir, exist_ok=True)
        experiment_id = time.time()
        
        with open(f"{self.model}_experiment_{experiment_id}.csv", "w", newline='') as res:
            writer = csv.writer(res)
            writer.writerow([
            "Layer", "Coefficients", "Coeff Norm", 
            "Explained Variance", "Accuracy", 
            "Confusion Matrix", "Classification Report"
        ])

        for result in self.cached_results:
                writer.writerow([
                    result["layer"],
                    json.dumps(result["coeffs"].tolist()),          # convert array to string
                    result["coeff_norm"],
                    json.dumps(result.get("explained_variance", [])),
                    result["accuracy"],
                    json.dumps(result["confusion_matrix"].tolist()),
                    json.dumps(result["classification_report"])
                ])

        print(f"Results saved to {save_dir}")
        return self.cached_results