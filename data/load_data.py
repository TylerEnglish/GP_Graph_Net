
#%%
from torch_geometric.datasets import QM9
from torch_geometric.data import DataLoader
#%% Load Dataset
dataset = QM9(root='data/QM9')
loader = DataLoader(dataset, batch_size=32, shuffle=True)

#%% Activate
for i in loader:
    print(i)
# %%
