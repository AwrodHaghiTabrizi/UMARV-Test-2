import dropbox
import os
import sys
import time
import re
import glob
import copy
import matplotlib.pyplot as plt
from getpass import getpass
import torch
from torch.utils.data import Dataset, DataLoader
from torchmetrics.classification import BinaryConfusionMatrix
from sklearn.model_selection import train_test_split

sys.path.append(f"{os.getenv('REPO_DIR')}/src")
from helpers import *

sys.path.append(f"{os.getenv('MODEL_DIR')}/src")
from dataset import *
from architecture import *

def set_device():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        print("Using the GPU! :)")
    else:
        print("Could not find GPU! :'( Using CPU only.")#\nIf you want to enable GPU, please to go Edit > Notebook Settings > Hardware Accelerator and select GPU.")
    return device

def initialize_model(device=None):
    if device is None:
        device = set_device()
    model = lane_model().to(device)
    return model

def create_datasets(device=None, datasets=None, benchmarks=None, include_all_datasets=False,
                    include_unity_datasets=False, include_real_world_datasets=False):

    dataset_dir = f"{os.getenv('ROOT_DIR')}/datasets"

    # Get data dirs from user specified datasets
    if datasets is not None:
        all_train_val_data_dirs = []
        for dataset in datasets:
            dataset_data_dirs = glob.glob(f"{dataset_dir}/{dataset}/data/*")
            all_train_val_data_dirs.extend(dataset_data_dirs)

    else:

        # Get data dirs from sample
        if not (include_all_datasets or include_unity_datasets or include_real_world_datasets):
            sample_dataset_dir = "sample/sample_dataset"
            dataset_data_dirs = glob.glob(f"{dataset_dir}/{sample_dataset_dir}/data/*")
            all_train_val_data_dirs.extend(dataset_data_dirs)

        # Get data dirs from all datasets
        else:
            all_train_val_data_dirs = []
            for dataset_category in ["unity", "real_world"]:
                # Check to skip the category if not requested
                if not include_all_datasets and (
                    (dataset_category == "unity" and not include_unity_datasets) or
                    (dataset_category == "real_world" and not include_real_world_datasets) ):
                    continue
                category_data_dirs = glob.glob(f'{dataset_dir}/{dataset_category}/*/data/*')
                all_train_val_data_dirs.extend(category_data_dirs)

    # Get train/val label directories
    all_train_val_label_dirs = [re.sub(r'\bdata\b', 'label', data_dir) for data_dir in all_train_val_data_dirs]

    # Split into train and val
    train_data_dirs, val_data_dirs, train_label_dirs, val_label_dirs = train_test_split(
        all_train_val_data_dirs, all_train_val_label_dirs, test_size=0.2, random_state=random.randint(1, 100)
    )

    # Get data dirs from user specified benchmarks
    if benchmarks is not None:
        all_benchmark_data_dirs = []
        for dataset in benchmarks:
            dataset_data_dirs = glob.glob(f"{dataset_dir}/{dataset}/data/*")
            all_benchmark_data_dirs.extend(dataset_data_dirs)
    
    # Get data dirs from all benchmarks
    else:
        all_benchmark_data_dirs = glob.glob(f'{dataset_dir}/benchmarks/*/data/*')

    # Get benchmark label directories
    all_benchmark_label_dirs = [re.sub(r'\bdata\b', 'label', data_dir) for data_dir in all_benchmark_data_dirs]

    # Create train/val/benchmark datasets
    train_dataset = Dataset_Class(
        data_dirs=train_data_dirs, label_dirs=train_label_dirs, device=device, label_input_threshold=.1
    )
    val_dataset = Dataset_Class(
        data_dirs=val_data_dirs, label_dirs=val_label_dirs, device=device, label_input_threshold=.1
    )
    benchmark_dataset = Dataset_Class(
        data_dirs=all_benchmark_data_dirs, label_dirs=all_benchmark_label_dirs, device=device, label_input_threshold=.1
    )

    return train_dataset, val_dataset, benchmark_dataset

def create_dataloaders(train_dataset, val_dataset, benchmark_dataset, batch_size=32, val_size=100):
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=val_size, shuffle=True)
    benchmark_dataloader = DataLoader(benchmark_dataset, batch_size=50, shuffle=False)
    return train_dataloader, val_dataloader, benchmark_dataloader



# Train the model for one batch
def train_model(model, criterion, optimizer, train_dataloader):
    model.train()
    _, data, label = next(iter(train_dataloader))
    optimizer.zero_grad()
    outputs = model(data)
    loss = criterion(outputs, label)
    loss.backward()
    optimizer.step()
    return loss.item()

# Validate the model
def validate_model(model, val_dataloader, device):
    model.eval()
    confusion_matrix = BinaryConfusionMatrix().to(device)
    TN_total = 0
    FP_total = 0
    FN_total = 0
    TP_total = 0

    with torch.no_grad():
        _, data, label = next(iter(val_dataloader))
        output = model(data)
        [TN, FP], [FN, TP] = confusion_matrix(output[:,:,:,1],label[:,:,:,1])
        TN_total += TN.item()
        FP_total += FP.item()
        FN_total += FN.item()
        TP_total += TP.item()

    accuracy = (TP_total + TN_total) / (TP_total + TN_total + FP_total + FN_total)

    if (TN_total+FP_total == 0):
        TN_rate = None
        FP_rate = None
    else:
        TN_rate = TN_total / (TN_total + FP_total)
        FP_rate = FP_total / (TN_total + FP_total)

    if (TP_total+FP_total == 0):
        precision = None
    else:
        precision = TP_total / (TP_total + FP_total)

    if (TP_total+FN_total == 0):
        recall = None
    else:
        recall = TP_total / (TP_total + FN_total)

    if(TP == 0):
        f1_score = 0
    else:
        f1_score = 2 * precision * recall / (precision + recall)

    #Add AUC - ROC Curve

    metrics = {
        'accuracy' : accuracy,
        'TN_rate' : TN_rate,
        'FP_rate' : FP_rate,
        'precision' : precision,
        'recall' : recall,
        'f1_score' : f1_score
    }

    return metrics

#The training loop
def training_loop(model, criterion, optimizer, train_dataloader, val_dataloader, device, num_epochs=50, auto_stop=True):
    loss_hist = []
    best_val_performance = {'accuracy':-1}

    for epoch in range(1, num_epochs+1):
        train_loss = train_model(model, criterion, optimizer, train_dataloader)
        loss_hist.append(train_loss)
        val_performance = validate_model(model, val_dataloader, device)

        if val_performance['accuracy'] > best_val_performance['accuracy']:
            best_val_performance = val_performance
            best_model = copy.deepcopy(model)
        
        #Add auto stop

        if (epoch == 1 or epoch % 5 == 0):
            print(f'Epoch: {epoch}/{num_epochs}   <>   Train Loss: {train_loss:.4f}   <>   Val Acc: {100*val_performance["accuracy"]:.2f}%')

    return best_model, loss_hist, best_val_performance

#Visualize the training loss
def visualize_loss(loss_hist):
    plt.figure()
    plt.plot(torch.tensor(loss_hist, device='cpu'))
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss History')
    plt.show()

#Visualize the output of the model
def visualize_results(model, dataset, device, output_threshold=.5):

    rand_indices = random.sample(range(len(dataset)), 5)
    
    fig, axs = plt.subplots(5, 4, figsize=(13, 25))

    for i, idx in enumerate(rand_indices):

        # Get output
        raw_data, data, label = dataset[idx]
        data = data.unsqueeze(0) #remove and not squeeze in plot?
        model_output = model(data)
        soft = nn.Softmax(dim=1)
        soft_output = soft(model_output)
        ones_output = soft_output[0,1,:,:]
        output = torch.zeros(ones_output.shape, device=device)
        output[ones_output > output_threshold] = 1

        # Plot
        axs[i, 0].imshow(raw_data.squeeze().permute(1, 2, 0).detach().cpu().numpy())
        axs[i, 0].set_title("Data")
        axs[i, 1].imshow(label.permute(1,2,0)[:,:,1].squeeze().detach().cpu().numpy(), cmap='gray')
        axs[i, 1].set_title("Label")
        axs[i, 2].imshow(ones_output.squeeze().detach().cpu().numpy(), cmap='gray')
        axs[i, 2].set_title("Soft Output")
        axs[i, 3].imshow(output.squeeze().detach().cpu().numpy(), cmap='gray')
        axs[i, 3].set_title("Treshold Output")



















def download_datasets_from_dropbox(
    dbx_access_token = None, use_thread = False, datasets = None,
    include_all_datasets = False, include_unity_datasets = False,
    include_real_world_datasets = False, include_benchmarks = False ):

    if dbx_access_token is None:
        dbx_access_token = getpass("Enter your DropBox access token: ")
    dbx = dropbox.Dropbox(dbx_access_token)
    
    dbx_datasets_dir = '/UMARV/ML/datasets'

    if datasets is not None:
        dataset_dirs = datasets

    else:
        
        if not (include_all_datasets or include_unity_datasets or include_real_world_datasets or include_benchmarks):
            dataset_dirs = ["sample/sample_dataset"]
            
    
        else:

            dataset_dirs = []

            for dataset_category in ["unity", "real_world", "benchmarks"]:

                # Check to skip the category if not requested
                if not include_all_datasets and (
                    (dataset_category == "unity" and not include_unity_datasets) or
                    (dataset_category == "real_world" and not include_real_world_datasets)
                ):
                    continue

                # Collect dataset directories in DropBox for category
                dataset_category_dir = f"{dbx_datasets_dir}/{dataset_category}"
                result = dbx.files_list_folder(dataset_category_dir)
                for entry in result.entries:
                    if isinstance(entry, dropbox.files.FolderMetadata):
                        found_dataset_dir = entry.path_display.lower().replace(dbx_datasets_dir.lower(),"")
                        dataset_dirs.append(found_dataset_dir)
                while result.has_more:
                    result = dbx.files_list_folder_continue(result.cursor)
                    for entry in result.entries:
                        if isinstance(entry, dropbox.files.FolderMetadata):
                            found_dataset_dir = entry.path_display.lower().replace(dbx_datasets_dir.lower(),"")
                            dataset_dirs.append(found_dataset_dir)

    # Download datasets
    for dataset_dir in dataset_dirs:

        copy_directory_from_dropbox(
            source_dir = f"{dbx_datasets_dir}/{dataset_dir}",
            destination_dir = f"{os.getenv('ROOT_DIR')}/datasets/{dataset_dir}",
            dbx_access_token = dbx_access_token,
            use_thread = use_thread
        )