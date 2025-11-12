import random
import csv
from typing import List, Tuple, Set, Dict

# Set random seed for reproducibility
random.seed(42)

# File path
folder = "./src/IR-Benchmark-Dataset/data_iid/amazon2014-book"

# ----------------------------------------------------------------------------- #
# 1. Read all files
# ----------------------------------------------------------------------------- #

def read_tsv(file_path: str) -> List[List[str]]:
    """Read TSV file, return data including header"""
    try:
        with open(file_path, 'r', newline='', encoding='utf-8') as f:
            reader = csv.reader(f, delimiter='\t')
            data = [row for row in reader]
        return data
    except FileNotFoundError:
        print(f"Warning: File {file_path} not found")
        return []

def read_tsv_without_header(file_path: str) -> List[Tuple[int, int]]:
    """Read TSV file, return integer data without header"""
    try:
        with open(file_path, 'r', newline='', encoding='utf-8') as f:
            reader = csv.reader(f, delimiter='\t')
            next(reader)  # Skip header
            data = [(int(row[0]), int(row[1])) for row in reader if len(row) >= 2]
        return data
    except FileNotFoundError:
        print(f"Warning: File {file_path} not found")
        return []
    except StopIteration:
        print(f"Warning: File {file_path} is empty")
        return []

print("Reading files...")
# Read mapping files (including header)
user_list = read_tsv(f"{folder}/user_list.tsv")
item_list = read_tsv(f"{folder}/item_list.tsv")

# Read interaction files (without header)
train_data = read_tsv_without_header(f"{folder}/train.tsv")
test_data = read_tsv_without_header(f"{folder}/test.tsv")

if not user_list or not item_list:
    print("Error: Mapping files are empty or do not exist")
    exit()

print(f"Original data: {len(user_list)-1} users, {len(item_list)-1} items")
print(f"{len(train_data)} training interactions, {len(test_data)} test interactions")

if len(user_list) <= 1:
    print("Error: No user data")
    exit()

# ----------------------------------------------------------------------------- #
# 2. Randomly sample 1/10 of users
# ----------------------------------------------------------------------------- #

# Get all user IDs (skip header)
all_users = [int(row[1]) for row in user_list[1:]]  # user_procid

# Randomly sample 1/10 of users
sample_size = max(1, len(all_users) // 10)
sampled_users = set(random.sample(all_users, sample_size))

print(f"\nSampled {len(sampled_users)} users ({len(sampled_users)/len(all_users)*100:.1f}% of total users)")

# ----------------------------------------------------------------------------- #
# 3. Filter interaction data, keep only interactions of sampled users
# ----------------------------------------------------------------------------- #

# Filter training and test sets
sampled_train = [(u, i) for u, i in train_data if u in sampled_users]
sampled_test = [(u, i) for u, i in test_data if u in sampled_users]

# Get actually used items
used_items = set(i for _, i in sampled_train) | set(i for _, i in sampled_test)

print(f"After filtering: {len(sampled_train)} training interactions, {len(sampled_test)} test interactions")
print(f"Actually used items: {len(used_items)}")

# ----------------------------------------------------------------------------- #
# 4. Remap IDs
# ----------------------------------------------------------------------------- #

# Create new mapping (maintain ID continuity)
new_user_map = {old_uid: new_uid for new_uid, old_uid in enumerate(sorted(sampled_users))}
new_item_map = {old_iid: new_iid for new_iid, old_iid in enumerate(sorted(used_items))}

# Remap interaction data
remapped_train = [(new_user_map[u], new_item_map[i]) for u, i in sampled_train]
remapped_test = [(new_user_map[u], new_item_map[i]) for u, i in sampled_test]

# Sort as required
remapped_train.sort(key=lambda x: (x[0], x[1]))
remapped_test.sort(key=lambda x: (x[0], x[1]))

# ----------------------------------------------------------------------------- #
# 5. Update user list and item list (maintain correspondence)
# ----------------------------------------------------------------------------- #

# Build old mapping (from original user_list and item_list)
old_user_map_dict = {row[0]: int(row[1]) for row in user_list[1:]}  # {user_orgid: user_procid}
old_item_map_dict = {row[0]: int(row[1]) for row in item_list[1:]}  # {item_orgid: item_procid}

# Reverse mapping (from user_procid to user_orgid)
old_user_map_dict_reverse = {v: k for k, v in old_user_map_dict.items()}
old_item_map_dict_reverse = {v: k for k, v in old_item_map_dict.items()}

# Build new user_list and item_list
# New format: [user_orgid, user_procid]
new_user_list = [[old_user_map_dict_reverse[old_uid], new_user_map[old_uid]] 
                 for old_uid in sampled_users]
new_user_list.sort(key=lambda x: x[1])  # Sort by new ID
new_user_list.insert(0, ['user_orgid', 'user_procid'])  # Add header

# New format: [item_orgid, item_procid]
new_item_list = [[old_item_map_dict_reverse[old_iid], new_item_map[old_iid]] 
                 for old_iid in used_items]
new_item_list.sort(key=lambda x: x[1])  # Sort by new ID
new_item_list.insert(0, ['item_orgid', 'item_procid'])  # Add header

# ----------------------------------------------------------------------------- #
# 6. Write to files
# ----------------------------------------------------------------------------- #

def write_tsv(data: List[List], file_path: str) -> None:
    """Write to TSV file"""
    with open(file_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f, delimiter='\t')
        writer.writerows(data)

print("\nWriting to files...")

# Write interaction files
write_tsv([(str(u), str(i)) for u, i in remapped_train], f"{folder}/train.tsv")
write_tsv([(str(u), str(i)) for u, i in remapped_test], f"{folder}/test.tsv")

# Write mapping files (including header)
write_tsv(new_user_list, f"{folder}/user_list.tsv")
write_tsv(new_item_list, f"{folder}/item_list.tsv")

# Write statistics
with open(f"{folder}/summary.txt", 'w', encoding='utf-8') as f:
    f.write(f"Sampled users: {len(sampled_users)}\n")
    f.write(f"Original users: {len(all_users)}\n")
    f.write(f"Sample ratio: {len(sampled_users)/len(all_users)*100:.1f}%\n")
    f.write(f"Training interactions: {len(remapped_train)}\n")
    f.write(f"Test interactions: {len(remapped_test)}\n")
    f.write(f"Used items: {len(used_items)}\n")

print("Sampling completed!")
print(f"\nSampling results:")
print(f"- Users: {len(new_user_list)-1} (sampled_user_list.tsv)")
print(f"- Items: {len(new_item_list)-1} (sampled_item_list.tsv)")
print(f"- Training interactions: {len(remapped_train)} (sampled_train.tsv)")
print(f"- Test interactions: {len(remapped_test)} (sampled_test.tsv)")