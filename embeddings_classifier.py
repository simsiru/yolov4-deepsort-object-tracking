import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torch.utils.data import Dataset
import torch.nn.functional as F
from utils import DBInterface
import numpy as np

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class EmbeddingsDataset(Dataset):
    def __init__(self, db_interface, embeddings_dir=None, embeddings_labels_dir=None, transforms=None, target_transforms=None):
        super(EmbeddingsDataset, self).__init__()

        self.idx_to_name_map = []
        self.num_samples_per_cls = None

        if embeddings_dir is not None or embeddings_labels_dir is not None:
            self.embeddings = torch.load(embeddings_dir)
            self.embeddings_labels = torch.load(embeddings_labels_dir)
        else:
            sql_script = """
            SELECT person_name, face_embedding
            FROM face_embeddings_and_depth_maps
            """
            df = db_interface.execute_sql_script(sql_script, return_result = True)
            first_arr = True
            for i, arr in enumerate(df['face_embedding']):
                np_arr = db_interface.bytes_to_numpy_array(arr)
                if first_arr:
                    self.embeddings = torch.tensor(np_arr)

                    self.embeddings_labels = torch.tensor([i])
                    self.num_samples_per_cls = self.embeddings.shape[0]
                    for j in range(1, self.num_samples_per_cls):
                        self.embeddings_labels = torch.cat((self.embeddings_labels, torch.tensor([i])), 0)

                    first_arr = False
                else:
                    self.embeddings = torch.cat((self.embeddings, torch.tensor(np_arr)), 0)

                    for j in range(self.num_samples_per_cls):
                        self.embeddings_labels = torch.cat((self.embeddings_labels, torch.tensor([i])), 0)

                self.idx_to_name_map.append(df['person_name'][i])

        np.save("face_embeddings_data/name_to_class_idx_map.npy", self.idx_to_name_map)

        self.transforms = transforms
        self.target_transforms = target_transforms

        self.n_classes = self.embeddings_labels[-1] + 1

    def __len__(self):
        return self.embeddings.shape[0]

    def __getitem__(self, idx):
        emb = self.embeddings[idx]
        label = self.embeddings_labels[idx]

        if self.transforms:
            emb = self.transforms(emb)

        if self.target_transforms:
            label = self.target_transforms(label)

        return emb, label



class EmbeddingsClassifier(nn.Module):
    def __init__(self, n_classes):
        super(EmbeddingsClassifier, self).__init__()

        self.l1 = nn.Linear(512, 256)
        self.l2 = nn.Linear(256, 256)
        self.l3 = nn.Linear(256, n_classes)

    def forward(self, x):
        x = self.l1(x)
        x = F.relu(x)
        x = self.l2(x)
        x = F.relu(x)
        x = self.l3(x)
        return x


""" from torchinfo import summary
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = EmbeddingsClassifier(3).to(device)

batch_size = 1
summary(model, input_size=(batch_size, 512)) """



def training(db_interface = None):
    if db_interface is None:
        db_interface = DBInterface(username='postgres', hostname='localhost',
        database='face_recognition', port_id=5432)

    transform = transforms.ToTensor()

    #train_dataset = EmbeddingsDataset("face_embeddings_data/face_embeddings.pt",
    #"face_embeddings_data/face_embeddings_labels.pt")
    train_dataset = EmbeddingsDataset(db_interface)

    train_dataloader = DataLoader(train_dataset, 16, shuffle=True)

    model = EmbeddingsClassifier(train_dataset.n_classes).to(DEVICE)

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.AdamW(model.parameters())

    n_epochs = 30

    for epoch in range(n_epochs):
        for emb, label in train_dataloader:
            emb = emb.to(DEVICE)
            label = label.to(DEVICE)

            output = model(emb)

            loss = criterion(output, label)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f'Epoch: {epoch}, Loss: {loss.item():.4f}')

    torch.save(model.state_dict(), 'embeddings_classifier_model/emb_classifier.pth')


if __name__ == "__main__":
    training()