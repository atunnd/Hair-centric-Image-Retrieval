import torch
import os
import torch
import torch.nn as nn
from tqdm import tqdm
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression

class Classifier:
    def __init__(self, model, train_loader, test_loader, args):
        self.model = model.to(args.device)
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.device = args.device
        self.mode = args.mode
        self.mode_model = args.model
        self.device = args.device

        if args.our_method:
            self.save_path = os.path.join(args.save_path, f"our_method_{self.mode}_{self.mode_model}")
        else:
            self.save_path = os.path.join(args.save_path, f"{self.mode}_{self.mode_model}")
        os.makedirs(self.save_path, exist_ok=True)
        self.training_features = []
        self.training_labels = []
        self.testing_features = []
        self.testing_labels = []

    
    def extracting_features(self):
        self.model.eval()

        with torch.no_grad():
            for batch in tqdm(self.train_loader, desc="Extracting training features"):
                images, labels = batch[0], batch[1]
                #print("images: ", images.shape)
                #print("label: ", labels.shape)
                x0= images
                x0 = x0.to(self.device)
                training_features = self.model.extract_features(x0)
                training_features = torch.nn.functional.normalize(training_features, dim=1)
                self.training_features.append(training_features.cpu())
                self.training_labels.append(labels)

        with torch.no_grad():
            for batch in tqdm(self.test_loader, desc="Extracting testing features"):
                images, labels = batch[0], batch[1]
                #print("images: ", images.shape)
                #print("label: ", labels.shape)
                x0 = images
                x0 = x0.to(self.device)
                testing_features = self.model.extract_features(x0)
                testing_features = torch.nn.functional.normalize(testing_features, dim=1)
                self.testing_features.append(testing_features.cpu())
                self.testing_labels.append(labels)
        
        self.training_features = torch.cat(self.training_features)
        self.training_labels = torch.cat(self.training_labels)
        self.testing_features = torch.cat(self.testing_features)
        self.testing_labels = torch.cat(self.testing_labels)

    def knn_eval(self, ks=(5, 10, 20, 27, 30, 40, 642)):
        print(f"Evaluating on KNN classifier with {self.device}")
        self.extracting_features()
        file_path = os.path.join(self.save_path, "knn_evaluation_results.txt")
        with open(file_path, "w") as f:
            f.write("KNN Evaluation Results\n")
            f.write("="*50 + "\n\n")

        for k in ks:
            knn = KNeighborsClassifier(n_neighbors=k, metric="cosine")
            knn.fit(self.training_features, self.training_labels)
            y_pred = knn.predict(self.testing_features)
            acc = accuracy_score(self.testing_labels, y_pred)
            report = classification_report(self.testing_labels, y_pred)
            cm = confusion_matrix(self.testing_labels, y_pred)

            with open(file_path, "a") as f:  # append thêm kết quả
                f.write(f"Results for k={k}\n")
                f.write("-"*40 + "\n")
                f.write(f"Accuracy: {acc:.4f}\n\n")
                f.write("Classification Report:\n")
                f.write(report + "\n\n")
                f.write("Confusion Matrix:\n")
                f.write(np.array2string(cm) + "\n\n")
                f.write("="*50 + "\n\n")

            print(f"✅ Appended results for k={k}")
        print(f"\n📂 All results saved in: {file_path}")
    
    
    def linear_probe_eval(self):
        print(f"Evaluating with Linear Probe on {self.device}")
        self.extracting_features()
        file_path = os.path.join(self.save_path, "linear_probe_results.txt")

        # Logistic regression (multi-class, one-vs-rest)
        clf = LogisticRegression(
            max_iter=5000, solver="lbfgs", multi_class="multinomial"
        )
        clf.fit(self.training_features, self.training_labels)
        y_pred = clf.predict(self.testing_features)

        acc = accuracy_score(self.testing_labels, y_pred)
        report = classification_report(self.testing_labels, y_pred)
        cm = confusion_matrix(self.testing_labels, y_pred)

        with open(file_path, "w") as f:
            f.write("Linear Probe Evaluation Results\n")
            f.write("="*50 + "\n\n")
            f.write(f"Accuracy: {acc:.4f}\n\n")
            f.write("Classification Report:\n")
            f.write(report + "\n\n")
            f.write("Confusion Matrix:\n")
            f.write(np.array2string(cm) + "\n\n")
            f.write("="*50 + "\n\n")

        print(f"✅ Linear probe results saved in: {file_path}")