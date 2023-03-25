import streamlit as st
import streamlit.components.v1 as components
import os

# Page 3: Code and Validation


def code_and_validation():
    st.write("# Code and Validation")
    st.write("In this section, we will discuss the code used in this app and how we validate the results.")
    st.write("## Code and what it does")
    st.write("The code used in this app is written in Python and uses several libraries, including PyTorch, RDKit, and NumPy. The code is used to calculate the dipole moment of a given chemical formula.")
    st.write("## Validating results")
    st.write("To validate the results, we compared the dipole moment calculated by the app with experimental data from the literature. The results were found to be in good agreement.")

<<<<<<< HEAD
# Load the dataset
dataset = ZINC(root='data/ZINC')

# Save the dataset as a file
with open('./data/dataset.pkl', 'wb') as f:
    pickle.dump(dataset, f)
    ''', 'python')

    st.subheader('How to load the data in new file')
    try:
        with open('./data/dataset.pkl', 'rb') as f:
            dataset = pickle.load(f)
    except FileNotFoundError:
        print("Error: dataset file not found.")
        exit(1)
    except Exception as e:
        print("Error loading dataset:", e)
        exit(1)
    st.code('''
try:
    with open('./data/dataset.pkl', 'rb') as f:
        dataset = pickle.load(f)
except FileNotFoundError:
    print("Error: dataset file not found.")
    exit(1)
except Exception as e:
    print("Error loading dataset:", e)
    exit(1)
    ''', 'python')
    st.subheader('Load in a graph')
    fig = plt.figure(figsize=(20, 15))
    st.code('''
fig = plt.figure(figsize=(20, 15))
data = dataset[n]
networkx_graph = to_networkx(data)
nx.draw_networkx(networkx_graph)
    ''', 'python')

    # Use a slider to select a graph from the dataset
    number = st.slider('Pick a number:', 0, len(dataset)-1)
    data = dataset[number]
    networkx_graph = to_networkx(data)
    nx.draw_networkx(networkx_graph)
    plt.show()
    st.pyplot(fig)
    st.header('GAE')
    st.code('''
class myEncoder(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(myEncoder, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = self.conv2(x, edge_index)
        return x

class myDecoder(torch.nn.Module):
    def __init__(self, hidden_dim, output_dim):
        super(myDecoder, self).__init__()
        self.linear1 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.linear2 = torch.nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = self.linear2(x)
        return x

class myGAE(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(myGAE, self).__init__()
        self.encoder = myEncoder(input_dim, hidden_dim)
        self.decoder = myDecoder(hidden_dim, output_dim)

    def forward(self, x, edge_index):
        x = self.encoder(x, edge_index)
        x = self.decoder(x)
        return x

    def encode(self, x, edge_index):
        return self.encoder(x, edge_index)
    ''','python')
    st.subheader('Training')
    st.code('''
for epoch in range(num_epochs):
    gae.train()  # set the model to train mode
    train_loss = 0
    for batch in train_loader:
        optimizer.zero_grad()
        outputs = gae(batch.x.float().to(device), batch.edge_index.to(device))
        x_hat = outputs
        loss = criterion(x_hat.float(), batch.x.float().to(device))
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * batch.num_graphs
    
    train_loss /= len(train_loader.dataset)
    
    gae.eval()  # set the model to evaluation mode
    val_loss = 0
    for batch in val_loader:
        with torch.no_grad():
            x_hat = gae(batch.x.float().to(device), batch.edge_index.to(device))
            loss = criterion(x_hat.float(), batch.x.float().to(device))
            val_loss += loss.item() * batch.num_graphs
    
    val_loss /= len(val_loader.dataset)

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(gae.state_dict(), './data/gae_model.pt')

    train_losses.append(train_loss)
    val_losses.append(val_loss)
    print(f"Epoch {epoch+1}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
    ''','python')
    st.header("GCN")
    st.code('''
class myGCNConv(MessagePassing):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(myGCNConv, self).__init__(aggr='add')
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        # Step 1: Add self-loops
        # edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))
        
        x = F.relu(self.conv1(x, edge_index))
        x = self.conv2(x, edge_index)

        # Step 3: Calculate the normalization
        row, col = edge_index
        deg = degree(row, x.size(0), dtype=x.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]

        # Step 4: Propagate the embeddings to the next layer
        x = self.propagate(edge_index, size=(x.size(0), x.size(0)), x=x, norm=norm)

        # Step 5: Apply global max pooling
        x = self.aggregate(x, torch.zeros(x.size(0), dtype=torch.long, device=device))

        # Step 6: Apply classifier
        # x = self.classifier(x)
        return x


    def message(self, x_j, norm):
        # Normalize node features.
        return norm.view(-1, 1) * x_j
    ''', 'python')

    st.subheader('Training')
    st.code('''
for epoch in range(num_epochs):
    gnn.train()
    train_loss = 0
    for graph in train_data:
        # print(batch)
        optimizer.zero_grad()
        x_hat = gnn(graph.x.float().to(device), graph.edge_index.to(device))
        # print("x_hat shape:", x_hat.shape)
        # print("batch.y shape:", batch.y.float().to(device).shape)
        

        loss = criterion(x_hat[0], graph.y.to(device))

        loss.backward()
        optimizer.step()
        train_loss += loss.item() #* batch.num_graphs
        # print("=", end="")
    
    train_loss /= len(train_data)
    print("")
    
    gnn.eval()
    val_loss = 0
    for graph in val_data:
        with torch.no_grad():
            x_hat = gnn(graph.x.float().to(device), graph.edge_index.to(device))
            loss = criterion(x_hat[0], graph.y.to(device))

            val_loss += loss.item() #* batch.num_graphs
            # print(">", end="")
    
    val_loss /= len(val_data)

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(gnn.state_dict(), './data/gcn_model.pt')

    train_losses.append(train_loss)
    val_losses.append(val_loss)
    print(f"Epoch {epoch+1}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
    scheduler.step()
    ''','python')


code_and_validation()
=======
code_and_validation()
>>>>>>> 0b1df013b95fff02d33a3a584431b9427780d07c
