import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
from torch_geometric.data import Data
from torch_geometric.nn import GATConv

def haversine(lat1, lon1, lat2, lon2):
    R = 6371
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
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

train_indices, test_indices = train_test_split(
    np.arange(len(df)), test_size=0.2, random_state=42
)

train_mask = torch.zeros(len(df), dtype=torch.bool)
test_mask = torch.zeros(len(df), dtype=torch.bool)
train_mask[train_indices] = True
test_mask[test_indices] = True

edges_train_only = []
edges_test = []
for i in range(len(coords)):
    for j in range(len(coords)):
        if i != j:
            dist = haversine(coords[i][0], coords[i][1], coords[j][0], coords[j][1])
            if dist <= distance_threshold:
                if i in train_indices and j in train_indices:
                    edges_train_only.append([i, j])
                elif i in test_indices or j in test_indices:
                    edges_test.append([i, j])

edge_index_train = torch.tensor(edges_train_only, dtype=torch.long).t()   # (2, train_edges)
edge_index_test = torch.tensor(edges_test, dtype=torch.long).t()    # (2, test_edges)

x = torch.tensor(
    df[["rating_scaled", "review_count_scaled"]].values,
    dtype=torch.float
)

graph_data = Data(
    x=x, 
    edge_index=edge_index_train, 
    train_mask=train_mask, 
    test_mask=test_mask
)

class GAT(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, heads=4, dropout=0.3):
        super(GAT, self).__init__()
        self.conv1 = GATConv(input_dim, hidden_dim, heads=heads, concat=True)
        self.dropout1 = torch.nn.Dropout(dropout)
        self.conv2 = GATConv(hidden_dim * heads, hidden_dim, heads=heads, concat=True)
        self.dropout2 = torch.nn.Dropout(dropout)
        self.conv3 = GATConv(hidden_dim * heads, output_dim, heads=1, concat=False)

    def forward(self, x, edge_index):
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

losses = []
mse_rating_values = []
mae_rating_values = []
mape_rating_values = []
mse_review_values = []
mae_review_values = []
mape_review_values = []

for epoch in range(180):
    model.train()
    optimizer.zero_grad()
    out = model(graph_data.x, graph_data.edge_index)
    loss = loss_fn(out[graph_data.train_mask], graph_data.x[graph_data.train_mask])
    loss.backward()
    optimizer.step()
    losses.append(loss.item())

    model.eval()
    with torch.no_grad():
        graph_data.edge_index = torch.cat([edge_index_train, edge_index_test], dim=1)
        pred = model(graph_data.x, graph_data.edge_index)

        pred_test = pred[graph_data.test_mask]
        true_test = graph_data.x[graph_data.test_mask]

        pred_rating_test = scalers["rating"].inverse_transform(
            pred_test[:, 0].detach().numpy().reshape(-1, 1)
        )
        true_rating_test = scalers["rating"].inverse_transform(
            true_test[:, 0].numpy().reshape(-1, 1)
        )
        pred_review_test = scalers["review_count"].inverse_transform(
            pred_test[:, 1].detach().numpy().reshape(-1, 1)
        )
        true_review_test = scalers["review_count"].inverse_transform(
            true_test[:, 1].numpy().reshape(-1, 1)
        )

        mse_rating = mean_squared_error(true_rating_test, pred_rating_test)
        mae_rating = mean_absolute_error(true_rating_test, pred_rating_test)
        mape_rating = mean_absolute_percentage_error(true_rating_test, pred_rating_test)

        mse_rating_values.append(mse_rating)
        mae_rating_values.append(mae_rating)
        mape_rating_values.append(mape_rating)

        mse_review = mean_squared_error(true_review_test, pred_review_test)
        mae_review = mean_absolute_error(true_review_test, pred_review_test)
        mape_review = mean_absolute_percentage_error(true_review_test, pred_review_test)

        mse_review_values.append(mse_review)
        mae_review_values.append(mae_review)
        mape_review_values.append(mape_review)

        graph_data.edge_index = edge_index_train

    print(f"Epoch {epoch+1}, Loss: {loss.item()}, MSE (Rating): {mse_rating}, "
          f"MAE (Rating): {mae_rating}, MAPE (Rating): {mape_rating}, "
          f"MSE (Review): {mse_review}, MAE (Review): {mae_review}, MAPE (Review): {mape_review}")

plt.figure(figsize=(15, 10))

plt.subplot(2, 2, 1)
plt.plot(losses, label="Лосс")
plt.xlabel("Эпоха")
plt.ylabel("Лосс")
plt.title("Лосс")
plt.legend()

plt.subplot(2, 2, 2)
plt.plot(mse_rating_values, label="MSE (рейтинг)")
plt.plot(mae_rating_values, label="MAE (рейтинг)")
plt.plot(mape_rating_values, label="MAPE (рейтинг)")
plt.xlabel("Эпоха")
plt.ylabel("Значение метрики")
plt.title("Метрики на тесте для рейтинга")
plt.legend()

plt.subplot(2, 2, 3)
# plt.plot(mse_review_values, label="MSE (Review)")
# plt.plot(mae_review_values, label="MAE (Review)")
plt.plot(mape_review_values, label="MAPE (Отзывы)")
plt.xlabel("Эпоха")
plt.ylabel("Значение метрики")
plt.title("Метрики на тесте для отзывов")
plt.legend()

plt.tight_layout()
plt.show()

