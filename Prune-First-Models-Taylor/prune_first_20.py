import torch
from torchvision.models import resnet18
from torchvision import datasets, models, transforms
import torch_pruning as tp
import time

# Set Variables
model = resnet18(pretrained=True)
num_features = model.fc.in_features 
model.fc = torch.nn.Linear(num_features, 9)
example_inputs = torch.randn(1, 3, 224, 224)
loss_function = torch.nn.CrossEntropyLoss()
device = "cuda" if torch.cuda.is_available() else "cpu"

# 1. Importance criterion
imp = tp.importance.GroupTaylorImportance() # or GroupNormImportance(p=2), GroupHessianImportance(), etc.

# 2. Ignore Output Layer for Pruning
ignored_layers = []
for m in model.modules():
    if isinstance(m, torch.nn.Linear) and m.out_features == 9:
        ignored_layers.append(m) # DO NOT prune the final classifier!

# 3. Create Meta Pruner Model
pruner = tp.pruner.MetaPruner( # We can always choose MetaPruner if sparse training is not required.
    model,
    example_inputs,
    importance=imp,
    pruning_ratio=0.2, # remove 50% channels, ResNet18 = {64, 128, 256, 512} => ResNet18_Half = {32, 64, 128, 256}
    ignored_layers=ignored_layers,
)

base_macs, base_nparams = tp.utils.count_ops_and_params(model, example_inputs)
with open("results_prune_first_20.txt", "w") as f:
    f.write(f"Base Params: {base_nparams/1e6}\n")
    f.write(f"Base MACs: {base_macs/1e6}\n")

# 3. Prune 
base_macs, base_nparams = tp.utils.count_ops_and_params(model, example_inputs)
if isinstance(imp, tp.importance.GroupTaylorImportance):
    outputs = model(example_inputs)
    target = torch.randint(0, 9, (1,))  # Assuming 1000 classes for classification
    loss = loss_function(outputs, target) 
    loss.backward() # before pruner.step()

pruner.step()
macs, nparams = tp.utils.count_ops_and_params(model, example_inputs)
with open("results_prune_first_20.txt", "a") as f:
    f.write(f"Pruned Params: {nparams/1e6}\n")
    f.write(f"Pruned MACs: {macs/1e6}\n")

# Finetune Model / Set Data Transformation and Data Loaders / Other variables
train_dir = "train"
val_dir = "val"

transforms_train = transforms.Compose([
    transforms.Resize((224, 224)),  
    transforms.RandomCrop((224, 224)),
    transforms.RandomRotation(10),
    transforms.RandomHorizontalFlip(), 
    transforms.ColorJitter(brightness=0.1),  
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) 
])

transforms_val = transforms.Compose([
    transforms.Resize((224, 224)),  
     transforms.CenterCrop((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


train_dataset = datasets.ImageFolder(train_dir, transforms_train)
val_dataset = datasets.ImageFolder(val_dir, transforms_val)

train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=12, shuffle=True, num_workers=0)
val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=12, shuffle=False, num_workers=0)

optimizer = torch.optim.AdamW(model.parameters(), lr=0.0001)

# Train model
train_loss=[]
train_accuary=[]
val_loss=[]
val_accuary=[]

num_epochs = 15   #(set no of epochs)
start_time = time.time() #(for showing time)
# Start loop

with open("results_prune_first_20.txt", "a") as f:
    for epoch in range(num_epochs): #(loop for every epoch)
        print("Epoch {} running".format(epoch)) #(printing message)
        """ Training Phase """
        model.train()    #(training model)
        running_loss = 0.   #(set loss 0)
        running_corrects = 0 
        # load a batch data of images
        for i, (inputs, labels) in enumerate(train_dataloader):
            inputs = inputs.to(device)
            labels = labels.to(device) 
            # forward inputs and get output
            optimizer.zero_grad()
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = loss_function(outputs, labels)
            # get loss value and update the network weights
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            running_corrects += torch.sum(preds == labels.data).item()
        epoch_loss = running_loss / len(train_dataset)
        epoch_acc = running_corrects / len(train_dataset) * 100.
        # Append result
        train_loss.append(epoch_loss)
        train_accuary.append(epoch_acc)
        # Print progress
        f.write('[Train #{}] Loss: {:.4f} Acc: {:.4f}% Time: {:.4f}s\n'.format(epoch+1, epoch_loss, epoch_acc, time.time() -start_time))

with open("results_prune_first_20.txt", "a") as f:
    model.eval()
    with torch.no_grad():
        running_loss = 0.
        running_corrects = 0
        for inputs, labels in val_dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = loss_function(outputs, labels)
            running_loss += loss.item()
            running_corrects += torch.sum(preds == labels.data).item()
        epoch_loss = running_loss / len(val_dataset)
        epoch_acc = running_corrects / len(val_dataset) * 100.
        # Append result
        val_loss.append(epoch_loss)
        val_accuary.append(epoch_acc)
        # Print progress
        f.write('[Validation #{}] Loss: {:.4f} Acc: {:.4f}% Time: {:.4f}s\n'.format(epoch+1, epoch_loss, epoch_acc, time.time()- start_time))

