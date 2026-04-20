#Machine Unlearning with SISA approach

import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset, TensorDataset

DATASET_PATH = "processed_data/dataset_bundle.npz"
CHECKPOINT_DIR = "checkpoints"
NUM_SHARDS = 15  
NUM_SLICES = 10   
BLACK_LABEL = "Black"


#Loading the training and testing set

def load_dataset():
    bundle = np.load(DATASET_PATH, allow_pickle=True)
    x_train = torch.tensor(bundle["X_train"])
    y_train = torch.tensor(bundle["y_train"])
    x_test = torch.tensor(bundle["X_test"])
    y_test = torch.tensor(bundle["y_test"])
    feature_names = bundle["feature_names"].tolist()

    train_dataset = TensorDataset(x_train, y_train)
    return train_dataset, x_train, y_train, x_test, y_test, feature_names


# The shallow NN model

class NNModel(nn.Module):
    def __init__(self, input_size, num_classes, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, 128), # input layer
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64), # hidden layer
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes), # output layer
        )

    def forward(self, x):
        return self.net(x)



# Splitting the dataset into K distinct shards .
def create_shards(dataset, num_shards):
    indices = np.random.permutation(len(dataset))
    shards = []
    for shard_idx in np.array_split(indices, num_shards):
        shards.append(Subset(dataset, shard_idx))
    return shards

# Splitting each shard into N slices 
def create_slices(shard_dataset, num_slices):
    slices = []
    for indices in np.array_split(np.arange(len(shard_dataset)), num_slices):
        slices.append(Subset(shard_dataset, indices))
    return slices



# Combining slices for training
def combine_slices(slices, removed_indices=None):
    if removed_indices is None:
        removed_indices = set()
    shard_dataset = slices[0].dataset
    indices = []
    for slice_dataset in slices:
        for local_index in slice_dataset.indices:
            original_index = int(shard_dataset.indices[local_index])
            if original_index not in removed_indices:
                indices.append(int(local_index))
    return Subset(shard_dataset, indices)


# black label fpr per shard
def compute_shard_black_fpr(model, cumulative_dataset, threshold=0.57):
    shard_dataset = cumulative_dataset.dataset
    X_list, y_list = [], []
    for local_index in cumulative_dataset.indices:
        x, y = shard_dataset[local_index]   
        X_list.append(x)
        y_list.append(int(y.item()))

    if len(X_list) == 0:
        return 0, 0

    X_t = torch.stack(X_list)
    y_np = np.array(y_list)
    model.eval()
    with torch.no_grad():
        logits = model(X_t)
        probs = torch.softmax(logits, dim=1)[:, 1].numpy()
    preds = (probs >= threshold).astype(int)
    bundle = np.load("processed_data/dataset_bundle.npz", allow_pickle=True)
    race_train = bundle["race_train"]  

    # Mapping original indices to race
    orig_indices = [int(shard_dataset.indices[li])
                    for li in cumulative_dataset.indices]
    race_np = race_train[orig_indices]
    black_neg_mask = (race_np == BLACK_LABEL) & (y_np == 0)
    n_black = int(black_neg_mask.sum())

    fp = int(((preds == 1) & black_neg_mask).sum())
    tn = int(((preds == 0) & black_neg_mask).sum())
    if (fp + tn) > 0:
        fpr_black = fp / (fp + tn) 
    else:
        fpr_black = 0
    return fpr_black, n_black


# Finding a balanced threshold for a shard slice if its Black fpr exceeds the target.
def find_fair_threshold(model, cumulative_dataset, fpr_target=0.25,
                        global_threshold=0.57):
    
    shard_dataset = cumulative_dataset.dataset
    X_list, y_list, orig_list = [], [], []
    for local_index in cumulative_dataset.indices:
        original_index = int(shard_dataset.indices[local_index])
        x, y = shard_dataset[local_index]   # local_index = position within shard
        X_list.append(x)
        y_list.append(int(y.item()))
        orig_list.append(original_index)

    if len(X_list) == 0:
        return global_threshold

    X_t = torch.stack(X_list)
    y_np = np.array(y_list)

    model.eval()
    with torch.no_grad():
        logits = model(X_t)
        probs = torch.softmax(logits, dim=1)[:, 1].numpy()
    base_acc = float((( probs >= global_threshold).astype(int) == y_np).mean())

    bundle = np.load("processed_data/dataset_bundle.npz", allow_pickle=True)
    race_train = bundle["race_train"]
    race_np = race_train[orig_list]
    black_neg = (race_np == BLACK_LABEL) & (y_np == 0)
    best_t = global_threshold
    best_fpr = float("inf")
    for t in np.linspace(global_threshold, 0.95, 40):
        preds_t  = (probs >= t).astype(int)
        acc_t = float((preds_t == y_np).mean())
        if acc_t < base_acc - 0.03:    # reject if accuracy drops > 3%
            continue
        fp_t = int(((preds_t == 1) & black_neg).sum())
        tn_t = int(((preds_t == 0) & black_neg).sum())
        fpr_t = fp_t / (fp_t + tn_t) if (fp_t + tn_t) > 0 else 0.0
        if fpr_t < best_fpr:
            best_fpr = fpr_t
            best_t = t
        if fpr_t <= fpr_target:
            break

    return float(best_t)


# Training the model sequentially on slice e1, e1 + e2, ..., e1 + e2 + ... + en.
def train_on_slices(model, slices, shard_id, start_slice_id= 0, 
                    previous_slices = None, removed_indices = None,
                    pos_weight = None, fairness_target = 0.25, 
                    global_threshold = 0.57, shard_thresholds = None):
    
    if pos_weight is not None:
        class_weights = torch.tensor([1.0, pos_weight], dtype=torch.float32)
        criterion = nn.CrossEntropyLoss(weight=class_weights)
    else:
        criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)

    if previous_slices is None:
        previous_slices = []
    if shard_thresholds is None:
        shard_thresholds = {}

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    for i, _ in enumerate(slices):
        slice_id = start_slice_id + i
        cumulative_dataset = combine_slices(
            previous_slices + slices[: i + 1], removed_indices
        )
        loader = DataLoader(cumulative_dataset, batch_size=32, shuffle=True)
        for _ in range(10):
            for x, y in loader:
                optimizer.zero_grad()
                pred = model(x)
                loss = criterion(pred, y)
                loss.backward()
                optimizer.step()

        # Checking the fairness per shard after training on all shard slices.
        fpr_black, n_black = compute_shard_black_fpr(
            model, cumulative_dataset, threshold=global_threshold)

        if n_black >= 5 and fpr_black > fairness_target:
            # Find a fairer threshold for this shard
            fair_t = find_fair_threshold(
                model, cumulative_dataset,
                fpr_target=fairness_target,
                global_threshold=global_threshold)
            shard_thresholds[shard_id] = fair_t
        elif n_black >= 5 and fpr_black <= fairness_target:
            # Already fair. Use global threshold.
            shard_thresholds[shard_id] = global_threshold

        path = f"{CHECKPOINT_DIR}/shard{shard_id}_slice{slice_id}.pt"
        torch.save(model.state_dict(), path)


# Training one model for each shard. 
def train_sisa(dataset, num_shards, num_slices, pos_weight=None,
               fairness_target=0.25, global_threshold=0.57):
    
    shards = create_shards(dataset, num_shards)
    models = []
    all_slices = []
    shard_thresholds = {}
    input_size = dataset.tensors[0].shape[1]
    for shard_id, shard in enumerate(shards):
        model = NNModel(input_size=input_size, num_classes=2)
        slices = create_slices(shard, num_slices)
        train_on_slices(
            model, slices, shard_id,
            pos_weight = pos_weight,
            fairness_target = fairness_target,
            global_threshold = global_threshold,
            shard_thresholds = shard_thresholds,
        )
        models.append(model)
        all_slices.append(slices)

    return models, shards, all_slices, shard_thresholds


#Aggregation

def aggregate_probabilities(models, x):
    preds = []
    for model in models:
        model.eval()
        with torch.no_grad():
            preds.append(torch.softmax(model(x), dim=1))
    return torch.mean(torch.stack(preds), dim=0)

# Aggregating the predictions from all models 
def aggregate_predict(models, x, threshold=0.5, shard_thresholds=None):
    if isinstance(x, torch.Tensor):
        x_t = x
    else: 
        x_t = torch.tensor(x, dtype=torch.float32)
    if shard_thresholds:
        votes = torch.zeros(x_t.shape[0])
        for shard_id, model in enumerate(models):
            model.eval()
            with torch.no_grad():
                probs = torch.softmax(model(x_t), dim=1)[:, 1]
            t = shard_thresholds.get(shard_id, threshold)
            votes += (probs >= t).float()
        return (votes >= len(models) / 2).long()
    avg_pred = aggregate_probabilities(models, x_t)
    if avg_pred.shape[1] == 2:
        return (avg_pred[:, 1] >= threshold).long()
    return torch.argmax(avg_pred, dim=1)

# Overall evaluation
def evaluate(models, x_test, y_test, threshold=0.5, shard_thresholds=None):
    predictions = aggregate_predict(
        models, x_test, threshold=threshold,
        shard_thresholds=shard_thresholds)
    accuracy = (predictions == y_test).float().mean().item()
    return accuracy


# Finding the location of datapoints

def find_example_locations(example_indices, shards, all_slices):
    locations = {}
    example_set = set(example_indices)

    for shard_id, shard in enumerate(shards):
        shard_indices_set = set(shard.indices)
        relevant_examples = example_set & shard_indices_set
        if not relevant_examples:
            continue
        for slice_id, slice_dataset in enumerate(all_slices[shard_id]):
            original_indices = {shard.indices[li] for li in slice_dataset.indices}
            # Found examples in a specific slice
            found = relevant_examples & original_indices
            if found:
                if shard_id not in locations:
                    locations[shard_id] = {
                        "slice_ids": set(),
                        "example_indices": set(),
                    }
                locations[shard_id]["slice_ids"].add(slice_id)
                locations[shard_id]["example_indices"].update(found)

    return locations


# Unlearning from a shard slice.
def unlearn_examples(shard_id, slices, removed_example_indices,
                     pos_weight=None, fairness_target=0.25,
                     global_threshold=0.57, shard_thresholds=None):
    
    sample_x, _ = slices[0][0]
    input_size = sample_x.shape[0]
    model = NNModel(input_size=input_size, num_classes=2)

    shard_dataset = slices[0].dataset
    affected_slice_ids = [] 
    for slice_id, slice_dataset in enumerate(slices):
        original_indices = [
            int(shard_dataset.indices[li]) for li in slice_dataset.indices
        ]
        if any(idx in removed_example_indices for idx in original_indices):
            affected_slice_ids.append(slice_id)
    start_slice_id = min(affected_slice_ids)
    # Loading the model from the slice before the unlearned slice
    if start_slice_id > 0:
        checkpoint = f"{CHECKPOINT_DIR}/shard{shard_id}_slice{start_slice_id - 1}.pt"
        model.load_state_dict(torch.load(checkpoint, map_location="cpu"))

    train_on_slices(
        model,
        slices[start_slice_id:],
        shard_id,
        start_slice_id = start_slice_id,
        previous_slices = slices[:start_slice_id],
        removed_indices = removed_example_indices,
        pos_weight = pos_weight,
        fairness_target = fairness_target,
        global_threshold = global_threshold,
        shard_thresholds = shard_thresholds,
    )

    return model