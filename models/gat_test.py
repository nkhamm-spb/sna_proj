import torch
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from torch_geometric.data import Data
from torch_geometric.nn import GATConv, global_mean_pool

def haversine(lat1, lon1, lat2, lon2):
    R = 6371 
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    return R * c

data_path = "table.csv"
df = pd.read_csv(data_path)

scalers = {
    "rating": StandardScaler(),
    "review_count": StandardScaler(),
}

df["rating_scaled"] = scalers["rating"].fit_transform(df[["rating"]])
df["review_count_scaled"] = scalers["review_count"].fit_transform(df[["review_count"]])


coords = df[["lat", "lon"]].values
distance_threshold = 0.5

edges = []
for i in range(len(coords)):
    for j in range(len(coords)):
        if i != j:
            dist = haversine(coords[i][0], coords[i][1], coords[j][0], coords[j][1])
            if dist <= distance_threshold:
                edges.append([i, j])

edge_index = torch.tensor(edges, dtype=torch.long).t()  # (2, num_edges)

x = torch.tensor(
    df[["rating_scaled", "review_count_scaled"]].values,
    dtype=torch.float
)

graph_data = Data(x=x, edge_index=edge_index)

class GAT(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, heads=4, dropout=0.3):
        super(GAT, self).__init__()
        self.conv1 = GATConv(input_dim, hidden_dim, heads=heads, concat=True)
        self.dropout1 = torch.nn.Dropout(dropout)
        self.conv2 = GATConv(hidden_dim * heads, hidden_dim, heads=heads, concat=True)
        self.dropout2 = torch.nn.Dropout(dropout)
        self.conv3 = GATConv(hidden_dim * heads, output_dim, heads=1, concat=False)

    def forward(self, x, edge_index, batch=None):
        x1 = torch.relu(self.conv1(x, edge_index))
        x1 = self.dropout1(x1)
        x2 = torch.relu(self.conv2(x1, edge_index))
        x2 = self.dropout2(x2 + x1)
        x3 = self.conv3(x2, edge_index)
        return x3

input_dim = 2
hidden_dim = 32
output_dim = 2
heads = 4
dropout = 0.2

model = GAT(input_dim, hidden_dim, output_dim, heads, dropout)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
loss_fn = torch.nn.MSELoss()

for epoch in range(100):
    model.train()
    optimizer.zero_grad()
    out = model(graph_data.x, graph_data.edge_index)
    loss = loss_fn(out, graph_data.x)
    loss.backward()
    optimizer.step()
    print(f"Epoch {epoch+1}, Loss: {loss.item()}")

model.eval()
with torch.no_grad():
    pred = model(graph_data.x, graph_data.edge_index)

pred_rating = scalers["rating"].inverse_transform(
    pred[:, 0].detach().numpy().reshape(-1, 1)
)
pred_review_count = scalers["review_count"].inverse_transform(
    pred[:, 1].detach().numpy().reshape(-1, 1)
)

df["predicted_rating"] = pred_rating # todo: обрезать сверху 5-кой
df["predicted_review_count"] = pred_review_count

print(df[["id", "name", "predicted_rating", "predicted_review_count"]].head(30))
