import json
import os
import numpy as np
import wget
import time
from ...ml.engine import ml_engine_adapter
import zipfile
from ...constants import FEDML_DATA_MNIST_URL
import logging
from scenarios.abstract_scenario import AbstractScenario
import torch
from torch.utils.data import DataLoader, TensorDataset, Subset

cwd = os.getcwd()

def partition_indices_dirichlet(dataset, num_clients, alpha, equal_samples):
    """
    Partitions the indices of the given dataset into num_clients parts.
    The partition is performed in a non-IID fashion via a Dirichlet distribution
    applied separately on each class. Finally, each client is downsampled
    to have exactly `equal_samples` indices (if possible).

    Assumes that the label is stored as the fourth element (index 3) in each sample.
    """
    # Get the labels as a NumPy array
    labels = dataset.tensors[3].cpu().numpy()  # adjust if needed (e.g., remove .cpu() if already numpy)
    classes = np.unique(labels)

    # Dictionary: client id -> list of indices from all classes
    client_indices = {i: [] for i in range(num_clients)}

    # For each class, partition its indices among clients via Dirichlet
    for c in classes:
        # Get all indices for class c
        idx_c = np.where(labels == c)[0]
        np.random.shuffle(idx_c)  # shuffle the order
        
        # Draw Dirichlet proportions for this class among all clients
        proportions = np.random.dirichlet(np.repeat(alpha, num_clients))
        # Compute how many indices of class c go to each client
        # (Using int conversion will usually be a rounding operation.)
        counts = (proportions * len(idx_c)).astype(int)
        # To ensure that we assign all indices of this class,
        # distribute any leftover indices randomly among clients.
        remainder = len(idx_c) - np.sum(counts)
        for _ in range(remainder):
            i = np.random.choice(num_clients)
            counts[i] += 1

        # Split and assign the indices for class c
        start_idx = 0
        for i in range(num_clients):
            end_idx = start_idx + counts[i]
            client_indices[i].extend(idx_c[start_idx:end_idx].tolist())
            start_idx = end_idx

    # Now, each client may have a different total number of indices.
    # We enforce that each client has exactly `equal_samples` samples.
    # (If a client gets fewer than equal_samples, you could choose to oversample;
    # here we simply use all available data in that client.)
    for i in range(num_clients):
        curr_count = len(client_indices[i])
        if curr_count > equal_samples:
            # Randomly choose equal_samples without replacement
            client_indices[i] = np.random.choice(client_indices[i], equal_samples, replace=False).tolist()
        else:
            # If a client has fewer samples than desired, you might decide to
            # leave it as is or oversample (here we leave it).
            client_indices[i] = client_indices[i]
        # (Optional) Shuffle the final list of indices for client i
        np.random.shuffle(client_indices[i])
        
    return client_indices
def load_data(args, train, test, dev):
    """
    Modified data loader that partitions the data among clients in a non-IID way
    (using a Dirichlet distribution per class) while ensuring that each client
    gets the same number of data points.
    
    Assumes:
      - args.client_num_in_total (e.g., 100)
      - args.batch_size
      - args.partition_alpha (Dirichlet parameter controlling heterogeneity)
      - The input objects (train, test, dev) have attributes g, w, x, y, z.
      
    For example, with 20,000 training samples and 100 clients,
    each client will receive 200 training samples.
    """
    num_clients = args.client_num_in_total

    # Create TensorDatasets (assume the label is the 4th tensor: train.y, etc.)
    train_dataset = TensorDataset(train.g, train.w, train.x, train.y, train.z)
    test_dataset = TensorDataset(test.g, test.w, test.x, test.y, test.z)
    dev_dataset = TensorDataset(dev.g, dev.w, dev.x, dev.y, dev.z)

    num_train_samples = len(train_dataset)
    num_test_samples = len(test_dataset)
    num_dev_samples = len(dev_dataset)

    # Determine the (equal) number of samples per client.
    # (If the total number isn’t divisible by the number of clients,
    # you may lose a few samples.)
    equal_train = num_train_samples // num_clients
    equal_test  = num_test_samples // num_clients
    equal_dev   = num_dev_samples // num_clients

    # Partition the indices by label using Dirichlet for heterogeneity.
    train_client_indices = partition_indices_dirichlet(train_dataset, num_clients, args.partition_alpha, equal_train)
    test_client_indices  = partition_indices_dirichlet(test_dataset, num_clients, args.partition_alpha, equal_test)
    dev_client_indices   = partition_indices_dirichlet(dev_dataset, num_clients, args.partition_alpha, equal_dev)

    # Create DataLoaders for each client using the computed subsets.
    train_data_local_dict = {}
    test_data_local_dict  = {}
    val_data_local_dict   = {}
    
    # Also record the number of samples per client.
    train_data_local_num_dict = {}
    test_data_local_num_dict  = {}
    val_data_local_num_dict   = {}

    for i in range(num_clients):
        train_subset = Subset(train_dataset, train_client_indices[i])
        test_subset  = Subset(test_dataset, test_client_indices[i])
        dev_subset   = Subset(dev_dataset, dev_client_indices[i])
        
        train_data_local_dict[i] = DataLoader(train_subset, batch_size=args.batch_size, shuffle=True)
        test_data_local_dict[i]  = DataLoader(test_subset, batch_size=args.batch_size, shuffle=True)
        val_data_local_dict[i]   = DataLoader(dev_subset, batch_size=args.batch_size, shuffle=True)

        train_data_local_num_dict[i] = len(train_client_indices[i])
        test_data_local_num_dict[i]  = len(test_client_indices[i])
        val_data_local_num_dict[i]   = len(dev_client_indices[i])

    return (train_data_local_dict,
            test_data_local_dict,
            val_data_local_dict,
            train_data_local_num_dict,
            test_data_local_num_dict,
            val_data_local_num_dict)
# def load_data(args, train, test, dev):
#     # Assuming args has attributes 'client_num' and 'batch_size'
#     clients_num = range(args.client_num_in_total)

#     # Creating TensorDatasets for train, test, and dev sets
#     train_dataset = TensorDataset(train.g, train.w, train.x, train.y, train.z)
#     test_dataset = TensorDataset(test.g, test.w, test.x, test.y, test.z)
#     dev_dataset = TensorDataset(dev.g, dev.w, dev.x, dev.y, dev.z)
#     num_train_samples = len(train_dataset)
#     num_test_samples = len(test_dataset)
#     num_dev_samples = len(dev_dataset)
#     min_samples_per_client = 5
#     available_train_samples = num_train_samples - args.client_num_in_total * min_samples_per_client
#     available_test_samples = num_test_samples - args.client_num_in_total * min_samples_per_client
#     available_dev_samples = num_dev_samples - args.client_num_in_total * min_samples_per_client

#     # Creating DataLoader for batching
#     # train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
#     # test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True)
#     # dev_loader = DataLoader(dev_dataset, batch_size=args.batch_size, shuffle=True)

#     # Dictionaries to hold the data for each client
#     train_data_local_dict = {user: [] for user in clients_num}
#     test_data_local_dict = {user: [] for user in clients_num}
#     val_data_local_dict = {user: [] for user in clients_num}
#     train_data_local_num_dict = {}
#     test_data_local_num_dict = {}
#     val_data_local_num_dict = {}
#     # np.random.seed(int(time.time()))
#     proportions_train = np.random.dirichlet([args.partition_alpha] * args.client_num_in_total)
#     proportions_test = np.random.dirichlet([args.partition_alpha] * args.client_num_in_total)
#     proportions_dev = np.random.dirichlet([args.partition_alpha] * args.client_num_in_total)

#     # Adjust proportions to consider the minimum samples reserved for each client
#     train_samples_per_client = (proportions_train * available_train_samples).astype(int) + min_samples_per_client
#     test_samples_per_client = (proportions_test * available_test_samples).astype(int) + min_samples_per_client
#     dev_samples_per_client = (proportions_dev * available_dev_samples).astype(int) + min_samples_per_client

#     # Re-adjust to ensure the exact number of samples is distributed
#     train_samples_per_client[-1] = num_train_samples - sum(train_samples_per_client[:-1])
#     test_samples_per_client[-1] = num_test_samples - sum(test_samples_per_client[:-1])
#     dev_samples_per_client[-1] = num_dev_samples - sum(dev_samples_per_client[:-1])

#     # Creating subsets and data loaders for each client
#     indices_train = np.random.permutation(num_train_samples)
#     indices_test = np.random.permutation(num_test_samples)
#     indices_dev = np.random.permutation(num_dev_samples)

#     # train_data_local_dict = {}
#     # test_data_local_dict = {}
#     # val_data_local_dict = {}

#     start = 0
#     for i in clients_num:
#         end = start + train_samples_per_client[i]
#         subset_indices = indices_train[start:end]
#         train_data_local_dict[i] = DataLoader(Subset(train_dataset, subset_indices), batch_size=args.batch_size, shuffle=True)
#         start = end

#     start = 0
#     for i in clients_num:
#         end = start + test_samples_per_client[i]
#         subset_indices = indices_test[start:end]
#         test_data_local_dict[i] = DataLoader(Subset(test_dataset, subset_indices), batch_size=args.batch_size, shuffle=True)
#         start = end

#     start = 0
#     for i in clients_num:
#         end = start + dev_samples_per_client[i]
#         subset_indices = indices_dev[start:end]
#         val_data_local_dict[i] = DataLoader(Subset(dev_dataset, subset_indices), batch_size=args.batch_size, shuffle=True)
#         start = end
#     # Distributing the batches among clients
#     # for i, (train_batch, test_batch, dev_batch) in enumerate(zip(train_loader, test_loader, dev_loader)):
#     #     user = i % args.client_num_in_total
#     #     train_data_local_dict[user].append(train_batch)
#     #     test_data_local_dict[user].append(test_batch)
#     #     val_data_local_dict[user].append(dev_batch)

    
#     # Calculate number of batches directly if dataset sizes and batch_size are known
# #     num_train_batches = (len(train_dataset) + args.batch_size - 1) // args.batch_size
# #     num_test_batches = (len(test_dataset) + args.batch_size - 1) // args.batch_size
# #     num_dev_batches = (len(dev_dataset) + args.batch_size - 1) // args.batch_size

# # # Generate Dirichlet distribution proportions
# #     proportions_train = np.random.dirichlet([args.partition_alpha] * args.client_num_in_total, 1).flatten()
# #     proportions_test = np.random.dirichlet([args.partition_alpha] * args.client_num_in_total, 1).flatten()
# #     proportions_dev = np.random.dirichlet([args.partition_alpha] * args.client_num_in_total, 1).flatten()

# # # Distribute batches based on calculated proportions and avoid early list conversion
# #     for i in clients_num:
# #       train_indices = np.random.choice(num_train_batches, int(proportions_train[i] * num_train_batches), replace=False)
# #       test_indices = np.random.choice(num_test_batches, int(proportions_test[i] * num_test_batches), replace=False)
# #       dev_indices = np.random.choice(num_dev_batches, int(proportions_dev[i] * num_dev_batches), replace=False)

# #     # Convert loaders to lists when needed and distribute
# #       all_train_batches = list(train_loader) if 'all_train_batches' not in locals() else all_train_batches
# #       all_test_batches = list(test_loader) if 'all_test_batches' not in locals() else all_test_batches
# #       all_dev_batches = list(dev_loader) if 'all_dev_batches' not in locals() else all_dev_batches

# #       for idx in train_indices:
# #         train_data_local_dict[i].append(all_train_batches[idx])
# #       for idx in test_indices:
# #         test_data_local_dict[i].append(all_test_batches[idx])
# #       for idx in dev_indices:
# #         val_data_local_dict[i].append(all_dev_batches[idx])


# #     # Calculating the number of samples for each client
#     for user in clients_num:
#         train_data_local_num_dict[user] = len(train_data_local_dict[user]) * args.batch_size
#         test_data_local_num_dict[user] = len(test_data_local_dict[user]) * args.batch_size
#         val_data_local_num_dict[user] = len(val_data_local_dict[user]) * args.batch_size

#     return (
#         train_data_local_dict,
#         test_data_local_dict,
#         val_data_local_dict,
#         train_data_local_num_dict,
#         test_data_local_num_dict,
#         val_data_local_num_dict
    # )


def download_mnist(data_cache_dir):
    if not os.path.exists(data_cache_dir):
        os.makedirs(data_cache_dir, exist_ok=True)

    file_path = os.path.join(data_cache_dir, "MNIST.zip")
    logging.info(file_path)

    # Download the file (if we haven't already)
    if not os.path.exists(file_path):
        wget.download(FEDML_DATA_MNIST_URL, out=file_path)

    file_extracted_path = os.path.join(data_cache_dir, "MNIST")
    if not os.path.exists(file_extracted_path):
        with zipfile.ZipFile(file_path, "r") as zip_ref:
            zip_ref.extractall(data_cache_dir)

def read_data(train_data_dir, test_data_dir):
    """parses data in given train and test data directories

    assumes:
    - the data in the input directories are .json files with
        keys 'users' and 'user_data'
    - the set of train set users is the same as the set of test set users

    Return:
        clients: list of non-unique client ids
        groups: list of group ids; empty list if none found
        train_data: dictionary of train data
        test_data: dictionary of test data
    """
    clients = []
    groups = []
    train_data = {}
    test_data = {}

    train_files = os.listdir(train_data_dir)
    train_files = [f for f in train_files if f.endswith(".json")]
    for f in train_files:
        file_path = os.path.join(train_data_dir, f)
        with open(file_path, "r") as inf:
            cdata = json.load(inf)
        clients.extend(cdata["users"])
        if "hierarchies" in cdata:
            groups.extend(cdata["hierarchies"])
        train_data.update(cdata["user_data"])

    test_files = os.listdir(test_data_dir)
    test_files = [f for f in test_files if f.endswith(".json")]
    for f in test_files:
        file_path = os.path.join(test_data_dir, f)
        with open(file_path, "r") as inf:
            cdata = json.load(inf)
        test_data.update(cdata["user_data"])

    clients = sorted(cdata["users"])

    return clients, groups, train_data, test_data


def batch_data(args, data, batch_size):

    """
    data is a dict := {'x': [numpy array], 'y': [numpy array]} (on one client)
    returns x, y, which are both numpy array of length: batch_size
    """
    data_x = data["x"]
    data_y = data["y"]

    # randomly shuffle data
    np.random.seed(100)
    rng_state = np.random.get_state()
    np.random.shuffle(data_x)
    np.random.set_state(rng_state)
    np.random.shuffle(data_y)

    # loop through mini-batches
    batch_data = list()
    for i in range(0, len(data_x), batch_size):
        batched_x = data_x[i : i + batch_size]
        batched_y = data_y[i : i + batch_size]
        batched_x, batched_y = ml_engine_adapter.convert_numpy_to_ml_engine_data_format(args, batched_x, batched_y)
        batch_data.append((batched_x, batched_y))
    return batch_data


def load_partition_data_mnist_by_device_id(batch_size, device_id, train_path="MNIST_mobile", test_path="MNIST_mobile"):
    train_path += os.path.join("/", device_id, "train")
    test_path += os.path.join("/", device_id, "test")
    return load_partition_data_mnist(batch_size, train_path, test_path)


def load_partition_data_mnist(
    args, batch_size
):
    scenario = AbstractScenario(filename="data/femnist_x/" + args.scenario_name + ".npz") 
    scenario.info()
    scenario.to_tensor()
    scenario.to_cuda()

    train = scenario.get_dataset("train")
    dev = scenario.get_dataset("dev")
    test = scenario.get_dataset("test")
    
    train_data_local_dict, test_data_local_dict,\
    val_data_local_dict, train_data_local_num_dict,\
    test_data_local_num_dict, val_data_local_num_dict = load_data(args, train, test, dev)

    return (
        args.client_num_in_total,
        train.y.shape[0],
        test.y.shape[0],
        dev.y.shape[0],
        train,
        test,
        dev,
        train_data_local_num_dict,
        train_data_local_dict,
        test_data_local_dict,
        val_data_local_dict,
        None,
    )
