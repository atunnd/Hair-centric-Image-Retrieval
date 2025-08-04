import torch
import random

class KMeans:
    def __init__(self, k=10, momentum=0.9):
        self.k = k
        self.momentum = momentum
        self.centroids = None

    def init_centroids(self, batch):
        """
        Randomly initialize k centroids from batch (tensor).
        """
        indices = torch.randperm(len(batch))[:self.k]
        self.centroids = batch[indices]

    def convergence_checked(self, old, new, tol=1e-4):
        """
        Check if centroids have converged.
        """
        return torch.all(torch.norm(old - new, dim=1) < tol)

    def assign_label(self, batch):
        """
        Assign each sample to the closest centroid (based on cosine similarity).
        """
        batch_norm = torch.nn.functional.normalize(batch, dim=1)
        centroids_norm = torch.nn.functional.normalize(self.centroids, dim=1)
        similarity = torch.matmul(batch_norm, centroids_norm.T)  # (N, k)
        labels = torch.argmax(similarity, dim=1)
        return labels

    def update_centroids(self, old_centroids, new_avg_centroids):
        return (1 - self.momentum) * old_centroids + self.momentum * new_avg_centroids

    def fit(self, batch, init=False, prev_centroids=None):
        """
        Fit KMeans on batch tensor of shape (N, D)
        """
        device = batch.device

        if init:
            self.init_centroids(batch)
        if prev_centroids:
            self.centroids = prev_centroids.to(device)

        for _ in range(10):  # max 10 iterations
            labels = self.assign_label(batch)
            new_centroids = torch.zeros_like(self.centroids)
            for i in range(self.k):
                mask = labels == i
                if mask.sum() == 0:
                    new_centroids[i] = self.centroids[i]  # giữ nguyên nếu cụm rỗng
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


class NegSampler(torch.nn.Module):
    def __init__(self, k=5, momentum=0.9):
        super(NegSampler, self).__init__()
        self.k = k
        self.momentum = momentum
        self.kmeans = KMeans(k=k, momentum=momentum)

    def neg_picker(self, centroids, batch):
        """
        Pick the 2nd most similar centroid (hard negative) for each sample.
        """
        batch_norm = torch.nn.functional.normalize(batch, dim=1)
        centroids_norm = torch.nn.functional.normalize(centroids, dim=1)
        sim = torch.matmul(batch_norm, centroids_norm.T)  # (N, k)
        top2 = torch.topk(sim, k=2, dim=1).indices  # (N, 2)
        neg_indices = top2[:, 1]  # index of 2nd most similar centroid
        negative = centroids[neg_indices]
        return negative

    def forward(self, embeddings, first_batch=False, prev_centroids=None):
        """
        batch: batch embeddings
        Return: centroids and negative batch (hard negative samples)
        """

        if first_batch:
            centroids = self.kmeans.fit(embeddings, init=True)
        else:
            centroids = self.kmeans.fit(embeddings, prev_centroids=prev_centroids)

        negatives = self.neg_picker(centroids, embeddings)
        return centroids, negatives
