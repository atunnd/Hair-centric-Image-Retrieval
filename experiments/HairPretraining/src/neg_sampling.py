import torch
import random
import numpy as np
from collections import defaultdict, Counter
import faiss


"""
class KMeans:
    def __init__(self, k=10, momentum=0.9):
        self.k = k
        self.momentum = momentum
        self.centroids = None

    def init_centroids(self, batch):
        indices = torch.randperm(len(batch))[:self.k]
        self.centroids = batch[indices]
        return self.centroids

    def convergence_checked(self, old, new, tol=1e-4):
        return torch.all(torch.norm(old - new, dim=1) < tol)

    def assign_label(self, batch, centroids):
        batch_norm = torch.nn.functional.normalize(batch, dim=1)
        centroids_norm = torch.nn.functional.normalize(centroids, dim=1)
        similarity = torch.matmul(batch_norm, centroids_norm.T)  # (N, k)
        labels = torch.argmax(similarity, dim=1)
        return labels

    def update_centroids(self, old_centroids, new_avg_centroids):
        return (1 - self.momentum) * old_centroids + self.momentum * new_avg_centroids

    def fit(self, batch, init=False, prev_centroids=None):
        device = batch.device

        if init:
            self.centroids = self.init_centroids(batch) # centroids 20
        if prev_centroids is not None:
            self.centroids = prev_centroids.to(device)

        for _ in range(10):  # max 10 iterations
            labels = self.assign_label(batch, self.centroids)
            new_centroids = torch.zeros_like(self.centroids)
            for i in range(self.k):
                mask = labels == i
                if mask.sum() == 0:
                    new_centroids[i] = self.centroids[i]  
                else:
                    new_centroids[i] = batch[mask].mean(dim=0)

            updated_centroids = self.update_centroids(self.centroids, new_centroids)
            if self.convergence_checked(self.centroids, updated_centroids):
                break
            self.centroids = updated_centroids

        return self.centroids

    @staticmethod
    def cosine_similarity(x, y):
        x = torch.nn.functional.normalize(x, dim=-1)
        y = torch.nn.functional.normalize(y, dim=-1)
        return torch.sum(x * y, dim=-1)


class NegSamplerMiniBatch(torch.nn.Module):
    def __init__(self, k=5, momentum=0.9):
        super(NegSamplerMiniBatch, self).__init__()
        self.k = k
        self.momentum = momentum
        self.kmeans = KMeans(k=k, momentum=momentum)

    def neg_picker(self, centroids, batch, real_batch):
        # Normalize for cosine similarity
        batch_norm = torch.nn.functional.normalize(batch, dim=1)
        centroids_norm = torch.nn.functional.normalize(centroids, dim=1)

        # Similarity between each batch sample and centroid
        sim = torch.matmul(batch_norm, centroids_norm.T)  # [B, C]
        
        # Get the 2nd most similar centroid index for each sample
        top2 = torch.topk(sim, k=2, dim=1).indices
        neg_indices = top2[:, 1]  # [B]
        
        # Get the corresponding centroid vectors
        negative_centroids = centroids[neg_indices]  # [B, D]
        negative_centroids_norm = torch.nn.functional.normalize(negative_centroids, dim=1)
        
        # Now, for each centroid, find the sample in batch that is closest (cosine sim)
        sim2 = torch.matmul(negative_centroids_norm, batch_norm.T)  # [B, B]
        
        # To avoid selecting itself as negative (optional but recommended)
        mask = torch.eye(sim2.size(0), device=sim2.device).bool()
        sim2.masked_fill_(mask, float('-inf'))

        # Get the index of the sample in batch closest to the centroid (but not itself)
        neg_sample_indices = torch.argmax(sim2, dim=1)  # [B]
        #negative_samples = batch[neg_sample_indices]  # [B, D]

        negative_samples = real_batch[neg_sample_indices]
        
        return negative_samples

    def forward(self, embeddings, real_batch, first_batch=False, prev_centroids=None):
        if first_batch:
            centroids = self.kmeans.fit(embeddings, init=True)
        else:
            centroids = self.kmeans.fit(embeddings, prev_centroids=prev_centroids)

        negatives = self.neg_picker(centroids, embeddings, real_batch)

        return centroids, negatives
"""

import torch
import faiss
import os


class Kmean_Faiss:
    def __init__(self, dim=128, k=15, device=0, momentum=0.9, save_path=""):
        self.dim = dim
        self.k = k
        self.device = device
        self.momentum = momentum  
        self.res = faiss.StandardGpuResources()

        self.kmeans = faiss.Clustering(dim, k)
        self.kmeans.niter = 25
        self.kmeans.verbose = True

        self.index_flat = faiss.IndexFlatL2(dim)
        self.index_flat = faiss.index_cpu_to_gpu(self.res, device, self.index_flat)

        self.save_path = os.path.join(save_path, "centroids")
        os.makedirs(self.save_path, exist_ok=True)


    def fit(self, embeddings: torch.Tensor, batch_id: int):
        # Convert to numpy for FAISS
        emb_np = embeddings.detach().cpu().numpy().astype("float32")
        self.kmeans.train(emb_np, self.index_flat)

        # Get centroids from FAISS and store them as torch tensor on GPU
        centroids_np = faiss.vector_float_to_array(self.kmeans.centroids) \
            .reshape(self.k, self.dim).astype("float32")

        self.centroids = torch.from_numpy(centroids_np).to(embeddings.device)

        # Update centroid with EMA
        centroid_path = os.path.join(self.save_path, f"centroid_{batch_id}")
        if os.path.exists(centroid_path):
            loading_path = centroid_path
            ema_centroids = torch.load(loading_path, map_location=self.device)
            self.centroids = self.momentum*ema_centroids + (1-self.momentum) *self.centroids
        
        # save new centroids
        torch.save(self.centroids, centroid_path)

    def query_hard_negative_centroid(self, embeddings: torch.Tensor, query_id: int):
        # Compute distances from the query sample to all centroids → pick top 2
        query_vec = embeddings[query_id].unsqueeze(0)  # shape: (1, dim)
        distances = torch.cdist(query_vec, self.centroids)  # shape: (1, k)
        top2_idx = torch.topk(distances, k=2, dim=1).indices
        return self.centroids[top2_idx[:, 1]]  # return the 2nd closest centroid

    def query_hard_negative_sample(self, embeddings: torch.Tensor, query_id: int):
        # Get the 2nd closest centroid
        negative_centroid = self.query_hard_negative_centroid(embeddings, query_id).unsqueeze(0)

        # Remove the query sample from the embeddings
        mask = torch.ones(embeddings.size(0), dtype=torch.bool, device=embeddings.device)
        mask[query_id] = False
        embeddings_filtered = embeddings[mask]

        # Find the farthest sample from the 2nd centroid
        distances = torch.cdist(negative_centroid, embeddings_filtered)  # shape: (1, N-1)
        negative_idx = torch.argmax(distances, dim=1)

        return embeddings_filtered[negative_idx]

class NegSamplerMiniBatch(torch.nn.Module):
    def __init__(self, k=5, dim=128, device="cuda", negative_centroid=True):
        super(NegSamplerMiniBatch, self).__init__()
        self.k = k
        self.dim = dim
        self.kmeans = Kmean_Faiss(dim, k, device)
        self.negative_centroid = negative_centroid
        self.query = self.kmeans.query_hard_negative_centroid if self.negative_centroid else self.kmeans.query_hard_negative_sample

    def forward(self, embeddings, batch_id):
        """
        batch: batch embeddings
        Return: hard negative sample/centroid
        """

        self.kmeans.fit(embeddings, batch_id)

        neg_embeddings = torch.zeros_like(embeddings) 
        for idx in range(len(embeddings)):
            neg_embeddings[idx] = self.query(embeddings, idx)
        
        return neg_embeddings

def NegSamplerClasses():
    pass

def NegSamplerRandomly(embeddings: torch.Tensor):
    """
    Randomly shuffle all samples in the batch embeddings.
    
    Args:
        embeddings (torch.Tensor): shape (batch_size, dim)
        
    Returns:
        shuffled_embeddings (torch.Tensor): shuffled embeddings
        permutation (torch.Tensor): indices of shuffle (to map original -> shuffled)
    """
    batch_size = embeddings.size(0)
    permutation = torch.randperm(batch_size, device=embeddings.device)
    shuffled_embeddings = embeddings[permutation]
    return shuffled_embeddings