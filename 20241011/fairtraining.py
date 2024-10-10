import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torchvision import transforms
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim import lr_scheduler
from sklearn.metrics import classification_report, f1_score, roc_auc_score, accuracy_score
from pathlib import Path

def calculate_distance_penalty(penalty_model, distances):
    
    
    diff = penalty_model.score_samples(np.array(distances).reshape(-1, 1))
    print("diff: ", len(diff))
    if len(diff) > 8:
        print("Num distances: ", len(distances))
        print(print("distances: ", distances))
    diff = 1 - diff
    diff = torch.tensor(diff)
    return diff

class CustomCrossEntropyLoss(nn.Module):
    def __init__(self, penalty_model):
        super(CustomCrossEntropyLoss, self).__init__()
        self.penalty_model = penalty_model
    def forward(self, outputs, targets, distances, device, epoch, total_epochs, start): # criterion
        ce_loss = F.cross_entropy(outputs, targets, reduction="none") # reduction="none"１つずつ距離を適用させるため
    
        if epoch <= start:
            return ce_loss.mean(), ce_loss.mean()
        else:
            penalty = calculate_distance_penalty(self.penalty_model, distances).to(device)
            #alpha = 0.95
            #print("penalty: ", penalty)
            #print("distance: ", distances)
            alpha = 1 #
            penalty = torch.softmax(penalty, dim=0)
            if len(penalty) > 8:
                print("penalty", penalty)
            return ce_loss.mean(), (ce_loss * penalty * alpha).sum()

class ModelTrainer:
    def __init__(self, penalty_model, model_save_directory):
        self.penalty_model = penalty_model
        self.model_save_directory = model_save_directory

    def train(self, model, train_loader, valid_loader, start, num_epochs=25, lr=1e-5, weight_decay=1e-4):
        
        train_losses = []
        valid_losses = []
        train_f1s = []
        valid_f1s = []
        train_accuracies = []
        valid_accuracies = []
        train_losses_nop = []
        valid_losses_nop = []

        best_val_loss = None
        best_val_file = None
    
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        criterion = CustomCrossEntropyLoss(self.penalty_model)
        optimizer = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    
        for epoch in range(num_epochs):
            model.train()
            running_loss = 0.0
            running_loss_nop = 0.0
            all_preds = []
            all_labels = []
            
            for inputs, labels, distances in train_loader:
                inputs, labels = inputs.to(device), labels.to(device) # data -> GPU
                optimizer.zero_grad()
                outputs = model(inputs)
  
                loss_nop, loss = criterion(outputs, labels, distances[:, 0], device, epoch, num_epochs, start)
                loss.backward() # Partial Derivative
                optimizer.step()
                
                running_loss += loss.item() * inputs.size(0)
                running_loss_nop += loss_nop.item() * inputs.size(0)
                all_preds.extend(outputs.detach().cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
            
            epoch_loss = running_loss / len(train_loader.dataset)
            epoch_loss_nop = running_loss_nop / len(train_loader.dataset)
            all_preds = np.argmax(all_preds, axis=1)
            epoch_f1 = f1_score(all_labels, all_preds, average='macro')
            epoch_acc = accuracy_score(all_labels, all_preds)
    
            train_losses.append(epoch_loss)
            train_losses_nop.append(epoch_loss_nop)
            train_f1s.append(epoch_f1)
            train_accuracies.append(epoch_acc)
            
            model.eval() # Validation
            val_loss = 0.0
            val_loss_nop = 0.0
            val_preds = []
            val_labels = []
            
            with torch.no_grad():
                for inputs, labels, distances in valid_loader:
                    inputs, labels = inputs.to(device), labels.to(device) # data -> GPU
                    outputs = model(inputs)
                    loss_nop, loss = criterion(outputs, labels, distances, device, epoch, num_epochs, start)
                    
                    val_loss += loss.item() * inputs.size(0)
                    val_loss_nop += loss_nop.item() * inputs.size(0)
                    val_preds.extend(outputs.cpu().numpy())
                    val_labels.extend(labels.cpu().numpy())

            if best_val_loss is None or best_val_loss > val_loss:
                best_val_loss = val_loss
                best_val_file = Path(self.model_save_directory) / f"model_{epoch}.pt"
                torch.save(model.state_dict(), best_val_file)
            
            val_loss /= len(valid_loader.dataset)
            val_loss_nop /= len(valid_loader.dataset)
            val_preds = np.argmax(val_preds, axis=1)
            val_f1 = f1_score(val_labels, val_preds, average='macro')
            val_acc = accuracy_score(val_labels, val_preds)
    
            valid_losses.append(val_loss)
            valid_losses_nop.append(val_loss_nop)
            valid_f1s.append(val_f1)
            valid_accuracies.append(val_acc)
    
            print(f'Validation Accuracy: {val_acc:.4f} | Loss: {val_loss:.4f} | F1: {val_f1:.4f}')
    
        epochs = range(num_epochs)
        plt.figure(figsize=(20, 4))

        fontsize=9
        
        plt.subplot(1, 4, 1)
        plt.plot(epochs, train_losses, label='Train Loss')
        plt.plot(epochs, valid_losses, label='Valid Loss')
        plt.xlabel('Epoch', fontsize=fontsize)
        plt.ylabel('Loss', fontsize=fontsize)
        plt.title('Loss', fontsize=fontsize)
        plt.legend(fontsize=fontsize)

        plt.subplot(1, 4, 2)
        plt.plot(epochs, train_losses_nop, label='Train Loss no penalty')
        plt.plot(epochs, valid_losses_nop, label='Valid Loss no penalty')
        plt.xlabel('Epoch', fontsize=fontsize)
        plt.ylabel('Loss no penalty', fontsize=fontsize)
        plt.title('Loss no penalty', fontsize=fontsize)
        plt.legend(fontsize=fontsize)
        
        plt.subplot(1, 4, 3)
        plt.plot(epochs, train_accuracies, label='Train Accuracy')
        plt.plot(epochs, valid_accuracies, label='Valid Accuracy')
        plt.xlabel('Epoch', fontsize=fontsize)
        plt.ylabel('Accuracy', fontsize=fontsize)
        plt.title('Accuracy', fontsize=fontsize)
        plt.legend(fontsize=fontsize)
        
        plt.subplot(1, 4, 4)
        plt.plot(epochs, train_f1s, label='Train F1 Score')
        plt.plot(epochs, valid_f1s, label='Valid F1 Score')
        plt.xlabel('Epoch', fontsize=fontsize)
        plt.ylabel('F1 Score', fontsize=fontsize)
        plt.title('F1 Score', fontsize=fontsize)
        plt.legend(fontsize=fontsize)

        plt.tight_layout()
        plt.show()
    
        return best_val_file

    def evaluate(self, model, valid_loader):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
        model.eval()
        with torch.no_grad():
            test_predictions = []
            test_outputs = []
            for inputs, labels, distance in valid_loader:
                inputs, labels = inputs.to(device), labels.to(device) # data -> GPU
                outputs = model(inputs)
                test_outputs.extend(outputs)
                test_predictions.extend(1 if x >= 0 else 0 for x in outputs)

        test_outputs = [o.cpu().item() for o in test_outputs]
        return test_predictions, test_outputs

    def report(self, df):
        print(classification_report(df["labels"], df["predictions"]))
    
        tones = df["skin tone"].unique()
        for t in tones:
            subset = df[df["skin tone"] == t]
            accuracy = accuracy_score(subset["labels"], subset["predictions"])
            print(f"Skin tone {t}: Accuracy {accuracy}")
        
