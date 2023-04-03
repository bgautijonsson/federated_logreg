import torch
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
import torch.optim as optim
import pyarrow.parquet as pq
import pandas as pd
import sys
from sklearn.metrics import classification_report

# Define the logistic regression model
# Define the logistic regression model with L2 regularization
class LogisticRegression(nn.Module):
    def __init__(self, input_size, output_size, reg_lambda=0.01):
        super(LogisticRegression, self).__init__()
        self.linear = nn.Linear(input_size, output_size)
        self.intercept = nn.Parameter(torch.zeros(1, output_size))
        self.reg_lambda = reg_lambda

    def forward(self, x):
        out = self.linear(x) + self.intercept
        out = nn.functional.softmax(out, dim=1)
        return out

    def l2_regularization(self):
        l2_reg = None
        for name, param in self.named_parameters():
            if "bias" not in name:
                if l2_reg is None:
                    l2_reg = param.norm(2) ** 2
                else:
                    l2_reg += param.norm(2) ** 2
        return l2_reg

    def loss_fn(self, outputs, targets):
        loss = nn.CrossEntropyLoss()(outputs, targets)
        l2_reg = self.l2_regularization()
        loss += self.reg_lambda * l2_reg
        return loss


# Define the dataset class
class FinancialDataset(Dataset):
    def __init__(self, data, labels):
        self.data = torch.FloatTensor(data)
        self.labels = torch.LongTensor(labels)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

# Define the training function
def train(model, dataloader, optimizer):
    model.train()
    running_loss = 0.0
    running_corrects = 0
    for inputs, labels in dataloader:
        optimizer.zero_grad()
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        loss = model.loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data)
    epoch_loss = running_loss / len(dataloader.dataset)
    epoch_acc = running_corrects.double() / len(dataloader.dataset)
    return epoch_loss, epoch_acc

# Define the test function with precision, recall, and F1 score for each model
def test(models, dataloaders_test):
    y_true_all = []
    y_pred_all = []
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for i in range(len(models)):
        models[i].eval()
        running_loss = 0.0
        running_corrects = 0
        y_true = []
        y_pred = []
        with torch.no_grad():
            for inputs, labels in dataloaders_test[i]:
                inputs = inputs.to(device)
                labels = labels.to(device)
                outputs = models[i](inputs)
                loss = models[i].loss_fn(outputs, labels)
                _, preds = torch.max(outputs, 1)
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
                y_true.extend(labels.cpu().numpy())
                y_pred.extend(preds.cpu().numpy())
        epoch_loss = running_loss / len(dataloaders_test[i].dataset)
        epoch_acc = running_corrects.double() / len(dataloaders_test[i].dataset)
        report = classification_report(y_true, y_pred, digits=4, output_dict=True, zero_division=0)
        epoch_precision = report['macro avg']['precision']
        epoch_recall = report['macro avg']['recall']
        epoch_f1 = report['macro avg']['f1-score']
        print('Test set loss for model {}: {:.4f}, accuracy: {:.4f}, precision: {:.4f}, recall: {:.4f}, f1-score: {:.4f}'.format(
            i, epoch_loss, epoch_acc, epoch_precision, epoch_recall, epoch_f1))
        y_true_all.extend(y_true)
        y_pred_all.extend(y_pred)
    report_all = classification_report(y_true_all, y_pred_all, digits=4, output_dict=True)
    epoch_precision_all = report_all['macro avg']['precision']
    epoch_recall_all = report_all['macro avg']['recall']
    epoch_f1_all = report_all['macro avg']['f1-score']
    print('Aggregate statistics: precision: {:.4f}, recall: {:.4f}, f1-score: {:.4f}'.format(
        epoch_precision_all, epoch_recall_all, epoch_f1_all))
    sys.stdout.flush()




# Define the main function
def main():
    # Define the hyperparameters
    learning_rate = 0.02
    num_epochs = 500
    batch_size = 256

    # Define the device (use CUDA if available)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Define the models for each financial institution
    models = []
    for i in range(30):
        models.append(LogisticRegression(5, 2).to(device))

    # Define the optimizer and criterion
    optimizers = [optim.SGD(model.parameters(), lr=learning_rate) for model in models]
    criterion = nn.CrossEntropyLoss()

    # Load the data and labels for each financial institution
    datasets_train = []
    datasets_test = []
    
    for i in range(30):
        data_file = f"train_data/data_{i}.parquet"
        data_table = pq.read_table(data_file)
        data_df = data_table.to_pandas()
        data_train = data_df.drop(columns=["ml"]).values
        labels_train = data_df["ml"].values
        data_test_file = f"test_data/test_data_{i}.parquet"
        data_test_table = pq.read_table(data_test_file)
        data_test_df = data_test_table.to_pandas()
        data_test = data_test_df.drop(columns=["ml"]).values
        labels_test = data_test_df["ml"].values
        
        dataset_train = FinancialDataset(data_train, labels_train)
        datasets_train.append(dataset_train)
        dataset_test = FinancialDataset(data_test, labels_test)
        datasets_test.append(dataset_test)
    
    dataloaders_train = []
    dataloaders_test = []
    for i in range(30):
        dataloader_train = torch.utils.data.DataLoader(datasets_train[i], batch_size=batch_size, shuffle=True)
        dataloaders_train.append(dataloader_train)
        dataloader_test = torch.utils.data.DataLoader(datasets_test[i], batch_size=batch_size, shuffle=False)
        dataloaders_test.append(dataloader_test)


    # Train the models using Federated Averaging
    for epoch in range(num_epochs):
        selected_models = torch.randperm(30)[:5] # select 5 random models
        global_grad = None
        for i in selected_models:
            optimizer = optimizers[i]
            model = models[i]
            dataloader = dataloaders_train[i]
            local_loss, local_acc = train(model, dataloader, optimizer)
            local_grad = [param.grad.data for param in model.parameters()]
            if global_grad is None:
                global_grad = local_grad
            else:
                global_grad = [global_grad[j] + local_grad[j] for j in range(len(local_grad))]
        global_grad = [grad / len(selected_models) for grad in global_grad]
        for i in range(i):
            model = models[i]
            optimizer = optimizers[i]
            for j, param in enumerate(model.parameters()):
                param.grad = global_grad[j]
            optimizer.step()
            optimizer.zero_grad()

        # Evaluate the global model on the test set
        test(models, dataloaders_test)

