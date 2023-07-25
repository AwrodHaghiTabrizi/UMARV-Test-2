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
from sklearn.metrics import confusion_matrix
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

def get_performance_metrics(TN_total, FP_total, FN_total, TP_total):

    print(f'{TN_total=}\n{FP_total=}\n{FN_total=}\n{TP_total=}')

    epsilon = 1e-8

    tn_rate = TN_total / (TN_total + FP_total + epsilon)
    fp_rate = FP_total / (TN_total + FP_total + epsilon)
    tp_rate = TP_total / (TP_total + FN_total + epsilon)
    fn_rate = FN_total / (TP_total + FN_total + epsilon)

    accuracy = (TN_total + TP_total) / (TN_total + FP_total + FN_total + TP_total)
    precision = TP_total / (TP_total + FP_total + epsilon)
    recall = TP_total / (TP_total + FN_total + epsilon)
    specificity = TN_total / (TN_total + FP_total + epsilon)
    f1_score = 2 * (precision * recall) / (precision + recall + epsilon)

    iou_lane = TP_total / (TP_total + FP_total + FN_total + epsilon)
    iou_background = TN_total / (TN_total + FP_total + FN_total + epsilon)
    m_iou = (iou_lane + iou_background) / 2

    total_pixels = TN_total + FP_total + FN_total + TP_total
    pixel_accuracy = (TN_total + TP_total) / (total_pixels + epsilon)
    mean_pixel_accuracy = (iou_lane + iou_background) / 2

    class_frequency = [TP_total + FN_total, TN_total + FP_total]

    fw_iou = (iou_lane * class_frequency[0] + iou_background * class_frequency[1]) / (total_pixels + epsilon)

    dice_coefficient = 2 * TP_total / (2 * TP_total + FP_total + FN_total + epsilon)

    boundary_f1_score = 2 * TP_total / (2 * TP_total + FP_total + FN_total + epsilon)

    metrics = {
        'TN Rate': tn_rate,
        'FP Rate': fp_rate,
        'TP Rate': tp_rate,
        'FN Rate': fn_rate,
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'Specificity': specificity,
        'F1 Score': f1_score,
        'IoU Lane': iou_lane,
        'IoU Background': iou_background,
        'Mean IoU': m_iou,
        'Pixel Accuracy': pixel_accuracy,
        'Mean Pixel Accuracy': mean_pixel_accuracy,
        'Frequency-Weighted IoU': fw_iou,
        'Dice Coefficient': dice_coefficient,
        'Boundary F1 Score': boundary_f1_score
    }

    # accuracy = (TP_total + TN_total) / (TP_total + TN_total + FP_total + FN_total)

    # if (TN_total+FP_total == 0):
    #     TN_rate = None
    #     FP_rate = None
    # else:
    #     TN_rate = TN_total / (TN_total + FP_total)
    #     FP_rate = FP_total / (TN_total + FP_total)

    # if (TP_total+FP_total == 0):
    #     precision = None
    # else:
    #     precision = TP_total / (TP_total + FP_total)

    # if (TP_total+FN_total == 0):
    #     recall = None
    # else:
    #     recall = TP_total / (TP_total + FN_total)

    # if(TP_total == 0):
    #     f1_score = 0
    # else:
    #     f1_score = 2 * precision * recall / (precision + recall)

    # #Add AUC - ROC Curve

    # metrics = {
    #     'accuracy' : accuracy,
    #     'TN_rate' : TN_rate,
    #     'FP_rate' : FP_rate,
    #     'precision' : precision,
    #     'recall' : recall,
    #     'f1_score' : f1_score
    # }

    return metrics

# Validate the model
def validate_model(model, dataloader, device):

    with torch.no_grad():
        model.eval()
        _, data, label = next(iter(dataloader))
        output = model(data)

        print(f'{output.shape=}')
        print(f'{label.shape=}\n')
        print(f'{output[0,0,0,0]=}')
        print(f'{output[0,1,0,0]=}\n')
        print(f'{label[0,0,0,0]=}')
        print(f'{label[0,1,0,0]=}\n')

        B, C, W, H = output.shape
        output = output.reshape(B * W * H, C)
        label = label.reshape(B * W * H, C)

        print(f'{B*W*H=}')
        print(f'{output.shape=}')
        print(f'{label.shape=}\n')

        output_binary = output.argmax(dim=1)
        label_binary = label.argmax(dim=1)

        print(f'{output_binary.shape=}')
        print(f'{label_binary.shape=}\n')
        print(f'{output_binary[0]=}')
        print(f'{label_binary[1]=}\n')

        conf_matrix = confusion_matrix(label_binary, output_binary)

        TN_total = conf_matrix[0, 0]
        FP_total = conf_matrix[0, 1]
        FN_total = conf_matrix[1, 0]
        TP_total = conf_matrix[1, 1]

        # soft = nn.Softmax(dim=3)
        # output_probabilities = soft(output)
        # threshold = 0.5
        # output_binary = (output_probabilities[:, :, :, 1] >= threshold).float()
        # confusion_mat = confusion_matrix(label.view(-1).cpu().numpy(), output_binary.view(-1).cpu().numpy())
        # TN, FP, FN, TP = confusion_mat.ravel()



        # # Assuming output and label are your prediction and ground truth tensors, respectively
        # softmax = nn.Softmax(dim=3)  # Create an instance of Softmax module with dim=3

        # output_probabilities = softmax(output)  # Applying softmax along the channel dimension (dim=3)

        # # Convert probabilities to binary predictions using thresholding (e.g., 0.5)
        # threshold = 0.5
        # output_binary = (output_probabilities[:, :, :, 1] >= threshold).float()

        # # Make sure the label tensor is already in binary format (0 or 1)

        # # Reshape output_binary to match the shape of the label tensor
        # output_binary = output_binary.view(-1).cpu().numpy()

        # confusion_mat = confusion_matrix(label.view(-1).cpu().numpy(), output_binary)

        # TN, FP, FN, TP = confusion_mat.ravel()




        # [TN, FP], [FN, TP] = confusion_matrix(output[:,:,:,1],label[:,:,:,1])
        # TN_total += TN.item()
        # FP_total += FP.item()
        # FN_total += FN.item()
        # TP_total += TP.item()
        
    metrics = get_performance_metrics(TN_total, FP_total, FN_total, TP_total)

    return metrics
    
# Validate the model
def test_model_on_benchmarks(model, device, all_benchmarks=True, benchmarks=None, visualize_the_results=True):
    model.eval()

    if all_benchmarks:
        all_benchmark_data_dirs = glob.glob(f'{os.getenv("ROOT_DIR")}/datasets/benchmarks/*/data/*.jpg')
        benchmarks = []
        for benchmark_data_dir in all_benchmark_data_dirs:
            # print(f"{benchmark_data_dir=}")
            # benchmark = benchmark_data_dir.split('/')[-3]
            benchmark = re.search(r'benchmark_\w+', benchmark_data_dir).group()
            # print(f"{benchmark=}")
            if benchmark not in benchmarks:
                benchmarks.append(benchmark)
    
    for benchmark in benchmarks:
        
        benchmark_data_dirs = glob.glob(f'{os.getenv("ROOT_DIR")}/datasets/benchmarks/{benchmark}/data/*')
        
        benchmark_label_dirs = [re.sub(r'\bdata\b', 'label', data_dir) for data_dir in benchmark_data_dirs]
        benchmark_dataset = Dataset_Class(
            data_dirs=benchmark_data_dirs, label_dirs=benchmark_label_dirs, device=device, label_input_threshold=.1
        )
        benchmark_dataloader = DataLoader(benchmark_dataset, batch_size=50, shuffle=False)

        # confusion_matrix = BinaryConfusionMatrix().to(device)
        # TN_total = 0
        # FP_total = 0
        # FN_total = 0
        # TP_total = 0

        with torch.no_grad():
            for _, data, label in benchmark_dataloader:

                output = model(data)

                # print(f'{output.shape=}')
                # print(f'{label.shape=}\n')
                # print(f'{output[0,0,0,0]=}')
                # print(f'{output[0,1,0,0]=}\n')
                # print(f'{label[0,0,0,0]=}')
                # print(f'{label[0,1,0,0]=}\n')

                B, C, W, H = output.shape
                output = output.reshape(B * W * H, C)
                label = label.reshape(B * W * H, C)

                # print(f'{B*W*H=}')
                # print(f'{output.shape=}')
                # print(f'{label.shape=}\n')

                output_binary = output.argmax(dim=1)
                label_binary = label.argmax(dim=1)

                # print(f'{output_binary.shape=}')
                # print(f'{label_binary.shape=}\n')
                # print(f'{output_binary[0]=}')
                # print(f'{label_binary[1]=}\n')

                conf_matrix = confusion_matrix(label_binary, output_binary)

                TN_total = conf_matrix[0, 0]
                FP_total = conf_matrix[0, 1]
                FN_total = conf_matrix[1, 0]
                TP_total = conf_matrix[1, 1]

                # # soft = nn.Softmax(dim=3)
                # # output_probabilities = soft(output)
                # # threshold = 0.5
                # # output_binary = (output_probabilities[:, :, :, 1] >= threshold).float()
                # # confusion_mat = confusion_matrix(label.view(-1).cpu().numpy(), output_binary.view(-1).cpu().numpy())
                # # TN, FP, FN, TP = confusion_mat.ravel()
                # # Assuming output and label are your prediction and ground truth tensors, respectively
                # softmax = nn.Softmax(dim=3)  # Create an instance of Softmax module with dim=3

                # output_probabilities = softmax(output)  # Applying softmax along the channel dimension (dim=3)

                # # Convert probabilities to binary predictions using thresholding (e.g., 0.5)
                # threshold = 0.5
                # output_binary = (output_probabilities[:, :, :, 1] >= threshold).float()

                # # Make sure the label tensor is already in binary format (0 or 1)

                # # Reshape output_binary to match the shape of the label tensor
                # output_binary = output_binary.view(-1).cpu().numpy()

                # confusion_mat = confusion_matrix(label.view(-1).cpu().numpy(), output_binary)

                # TN, FP, FN, TP = confusion_mat.ravel()
                # # [TN, FP], [FN, TP] = confusion_matrix(output[:, :, :, 1], label[:, :, :, 1])
                # TN_total += TN.item()
                # FP_total += FP.item()
                # FN_total += FN.item()
                # TP_total += TP.item()

        metrics = get_performance_metrics(TN_total, FP_total, FN_total, TP_total)

        print(f'{benchmark} metrics:')
        for metric in metrics:
            print(f'\t{metric}: {metrics[metric]:.4f}')

        visualize_results(model, benchmark_dataset, device, output_threshold=.5, num_samples=2)

    return

#The training loop
def training_loop(model, criterion, optimizer, train_dataloader, val_dataloader, device, num_epochs=50, auto_stop=True, auto_stop_patience=10):

    loss_hist = []
    best_val_performance = {'accuracy':-1}
    epochs_since_best_val_performance = 0

    for epoch in range(1, num_epochs+1):
        train_loss = train_model(model, criterion, optimizer, train_dataloader)
        loss_hist.append(train_loss)
        val_performance = validate_model(model, val_dataloader, device)

        if epoch == 1 or val_performance['Accuracy'] > best_val_performance['Accuracy']:
            best_val_performance = val_performance
            best_model = copy.deepcopy(model)
        else:
            epochs_since_best_val_performance += 1
        
        if auto_stop and epochs_since_best_val_performance >= auto_stop_patience:
            print(f'Epoch: {epoch}/{num_epochs}   <>   Train Loss: {train_loss:.4f}   <>   Val Acc: {100*val_performance["Accuracy"]:.2f}%')
            print(f'Training auto stopped. No improvement in validation accuracy for {auto_stop_patience} epochs.')
            break

        if (epoch == 1 or epoch % 5 == 0):
            print(f'Epoch: {epoch}/{num_epochs}   <>   Train Loss: {train_loss:.4f}   <>   Val Acc: {100*val_performance["Accuracy"]:.2f}%')

    return best_model, loss_hist, best_val_performance

def visualize_loss(loss_hist):
    plt.figure()
    plt.plot(torch.tensor(loss_hist, device='cpu'))
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss History')
    plt.show()

def visualize_results(model, dataset, device, output_threshold=.5, num_samples=5):

    rand_indices = random.sample(range(len(dataset)), num_samples)

    fig, axs = plt.subplots(num_samples, 4, figsize=(13, 5*num_samples))

    for i, idx in enumerate(rand_indices):
        raw_data, data, label = dataset[idx]
        data = data.unsqueeze(0)
        model_output = model(data)
        soft = nn.Softmax(dim=1)
        soft_output = soft(model_output)
        ones_output = soft_output[0,1,:,:]
        output = torch.zeros(ones_output.shape, device=device)
        output[ones_output > output_threshold] = 1

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