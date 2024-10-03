import torch
import torch.nn as nn
from torchvision import transforms
from torch.optim import Adam
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, f1_score, roc_auc_score, accuracy_score
import imageutils
from pathlib import Path

class ModelTrainer:
    def train(self, model, train_loader, valid_loader, model_save_directory, num_epochs=25, lr=1e-5, weight_decay=1e-4):

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        train_losses = []
        valid_losses = []
        train_f1s = []
        valid_f1s = []
        train_aucs = []
        valid_aucs = []
        train_accuracies = []
        valid_accuracies = []
    
        best_val_loss = None
        best_val_file = None
        
        criterion = nn.CrossEntropyLoss()
        optimizer = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    
        for epoch in range(num_epochs):
            model.train()
            running_loss = 0.0
            all_preds = []
            all_labels = []
            
            for i, (inputs, labels, _) in enumerate(train_loader):
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad() # 移動幅を０に設定している
                outputs = model(inputs)

                print("Outputs shape:", outputs.shape)
                print("Outputs:", outputs)
                print("Labels shape:", labels.shape)
                print("Labels:", labels)
                #print("Labels dtype:", labels.dtype)
    
                loss = criterion(outputs, labels)
                loss.backward() # 偏微分の実行。評価のときは実行しない。
                optimizer.step() # 偏微分に基づいて実際に移動する。評価のときは移動しない。評価のときは学習しないから。
                
                running_loss += loss.item() * inputs.size(0)
                all_preds.extend(outputs.detach().cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

            
            epoch_loss = running_loss / len(train_loader.dataset)
            all_preds = np.argmax(all_preds, axis=1) #一括処理
            epoch_f1 = f1_score(all_labels, all_preds)
            epoch_acc = accuracy_score(all_labels, all_preds)
            epoch_auc = roc_auc_score(all_labels, all_preds)
    
            train_accuracies.append(epoch_acc)
            train_losses.append(epoch_loss)
            train_f1s.append(epoch_f1)
            train_aucs.append(epoch_auc)
            
            model.eval() # Validation だから。
            val_loss = 0.0
            val_preds = []
            val_labels = []
    
            with torch.no_grad(): # Withの内側では偏微分をしません。
                for i, (inputs, labels, _) in enumerate(valid_loader):
                    inputs, labels = inputs.to(device), labels.to(device) # data -> GPU
                    outputs = model(inputs)
                    loss = criterion(outputs, labels) # バッチのデフォルトは平均値がかえってくる。
                    # no_grad()があるかないかで挙動が変わる。勾配計算(=偏微分)のための情報を保持しない。メモリ使用量が変わる。
                    # バッチサイズは学習と評価で違ってOK
                    val_loss += loss.item() * inputs.size(0)
                    val_preds.extend(outputs.cpu().numpy())
                    val_labels.extend(labels.cpu().numpy())
    
            if best_val_loss is None or best_val_loss > val_loss:
                best_val_loss = val_loss
                best_val_file = f"{Path(self.model_save_directory)}model_{epoch}.pt"
                torch.save(model.state_dict(), best_val_file)
            
            val_loss /= len(valid_loader.dataset)
            val_preds = np.argmax(val_preds, axis=1)
            val_f1 = f1_score(val_labels, val_preds)
            val_acc = accuracy_score(val_labels, val_preds)
            val_auc = roc_auc_score(val_labels, val_preds)
    
            valid_losses.append(val_loss)
            valid_accuracies.append(val_acc)
            valid_f1s.append(val_f1)
            valid_aucs.append(val_auc)
            
            print(f'Validation Accuracy: {val_acc:.4f} | Loss: {val_loss:.4f} | F1: {val_f1:.4f}')
    
        #torch.save(model.state_dict(), Path(model_save_directory) / f"model_last.pt")

        
        plt.figure(figsize=(16, 4))
        epochs = range(num_epochs)
        fontsize=9
        
        plt.subplot(1, 4, 1)
        plt.plot(epochs, train_losses, label='Train Loss')
        plt.plot(epochs, valid_losses, label='Valid Loss')
        plt.xlabel('Epoch', fontsize=fontsize)
        plt.ylabel('Loss', fontsize=fontsize)
        plt.title('Loss', fontsize=fontsize)
        plt.legend(fontsize=fontsize)
        
        plt.subplot(1, 4, 2)
        plt.plot(epochs, train_accuracies, label='Train Accuracy')
        plt.plot(epochs, valid_accuracies, label='Valid Accuracy')
        plt.xlabel('Epoch', fontsize=fontsize)
        plt.ylabel('Accuracy', fontsize=fontsize)
        plt.title('Accuracy', fontsize=fontsize)
        plt.legend(fontsize=fontsize)
        
        plt.subplot(1, 4, 3)
        plt.plot(epochs, train_f1s, label='Train F1 Score')
        plt.plot(epochs, valid_f1s, label='Valid F1 Score')
        plt.xlabel('Epoch', fontsize=fontsize)
        plt.ylabel('F1 Score', fontsize=fontsize)
        plt.title('F1 Score', fontsize=fontsize)
        plt.legend(fontsize=fontsize)
    
        plt.subplot(1, 4, 4)
        plt.plot(epochs, train_aucs, label='Train AUC Score')
        plt.plot(epochs, valid_aucs, label='Valid AUC Score')
        plt.xlabel('Epoch', fontsize=fontsize)
        plt.ylabel('AUC Score', fontsize=fontsize)
        plt.title('AUC Score', fontsize=fontsize)
        plt.legend(fontsize=fontsize)
        
        plt.tight_layout()
        plt.show()

        return best_val_file

    def evaluate(self, model, valid_loader):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
        model.eval() # 評価モードに入る。dropoutは学習にしか使わない。dropoutはノイズを入れている。ノイズに耐えられるように。
        with torch.no_grad():
            test_predictions = []
            test_outputs = []
            for inputs, labels, _ in valid_loader:
                inputs, labels = inputs.to(device), labels.to(device) # data -> GPU
                outputs = model(inputs)
                _, prediction = torch.max(outputs, 1) # outputsの大きい方が選ばれるけど、二値分類なので異なる
                test_predictions.extend(prediction) # Appendは１つ、extendは複数
                test_outputs.extend(outputs)
        
        test_outputs = [o.cpu().item() for o in test_outputs]
        print(classification_report(labels, test_predictions))
        return test_predictions, test_outputs