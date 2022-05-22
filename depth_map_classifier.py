import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torch.utils.data import Dataset
import torch.nn.functional as F
from utils import DBInterface
import numpy as np

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class DepthMapsDataset(Dataset):
    def __init__(self, db_interface, depth_maps=None, embeddings_labels_dir=None, transforms=None, target_transforms=None):
        super(DepthMapsDataset, self).__init__()

        self.idx_to_name_map = []
        self.num_samples_per_cls = None

        if depth_maps is not None or embeddings_labels_dir is not None:
            self.depth_maps = torch.load(depth_maps)
            self.embeddings_labels = torch.load(embeddings_labels_dir)
        else:
            sql_script = """
            SELECT person_name, face_depth_map
            FROM face_embeddings_and_depth_maps
            """
            df = db_interface.execute_sql_script(sql_script, return_result = True)
            first_arr = True
            for i, arr in enumerate(df['face_depth_map']):
                np_arr = db_interface.bytes_to_numpy_array(arr)
                if first_arr:
                    self.depth_maps = torch.tensor(np_arr)

                    self.embeddings_labels = torch.tensor([i])
                    self.num_samples_per_cls = self.depth_maps.shape[0]
                    for j in range(1, self.num_samples_per_cls):
                        self.embeddings_labels = torch.cat((self.embeddings_labels, torch.tensor([i])), 0)

                    first_arr = False
                else:
                    self.depth_maps = torch.cat((self.depth_maps, torch.tensor(np_arr)), 0)

                    for j in range(self.num_samples_per_cls):
                        self.embeddings_labels = torch.cat((self.embeddings_labels, torch.tensor([i])), 0)

                self.idx_to_name_map.append(df['person_name'][i])

        np.save("face_embeddings_data/name_to_class_idx_map.npy", self.idx_to_name_map)

        self.transforms = transforms
        self.target_transforms = target_transforms

        self.n_classes = self.embeddings_labels[-1] + 1
        self.depth_dims = (self.depth_maps.shape[1], self.depth_maps.shape[2])

    def __len__(self):
        return self.depth_maps.shape[0]

    def __getitem__(self, idx):
        dm = self.depth_maps[idx]
        label = self.embeddings_labels[idx]

        if self.transforms:
            dm = self.transforms(dm.unsqueeze(0))

        if self.target_transforms:
            label = self.target_transforms(label)

        return dm, label


class DepthMapsClassifier(nn.Module):
    def __init__(self, n_classes, depth_dims):
        super(DepthMapsClassifier, self).__init__()

        self.conv1 = nn.Conv2d(1, 8, 5)
        self.conv2 = nn.Conv2d(8, 16, 5)
        self.conv3 = nn.Conv2d(16, 32, 5)
        self.conv4 = nn.Conv2d(32, 64, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.flatten = nn.Flatten()

        lin_input = self.conv(torch.rand(1, 1, depth_dims[0], depth_dims[1]))
        #print(lin_input.shape)
        #print(lin_input.view(-1, lin_input.shape[1]*lin_input.shape[2]*lin_input.shape[3]).shape)
        #print(self.flatten(lin_input).shape)

        self.l1 = nn.Linear(self.flatten(lin_input).shape[1], 256)
        self.l2 = nn.Linear(256, 256)
        self.l3 = nn.Linear(256, n_classes)

    def conv(self, x):
        x = self.conv1(x)
        x = self.pool(F.relu(x))
        x = self.conv2(x)
        x = self.pool(F.relu(x))
        x = self.conv3(x)
        x = self.pool(F.relu(x))
        x = self.conv4(x)
        x = self.pool(F.relu(x))
        return x

    def forward(self, x):
        x = self.conv(x)

        x = self.flatten(x)

        x = self.l1(x)
        x = self.l2(F.relu(x))
        x = self.l3(F.relu(x))

        return x


""" from torchinfo import summary
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = DepthMapsClassifier(3, (320, 280)).to(device)

batch_size = 1
summary(model, input_size=(batch_size, 1, 320, 280)) """






def training(db_interface = None):
    if db_interface is None:
        db_interface = DBInterface(username='postgres', hostname='localhost',
        database='face_recognition', port_id=5432)

    def rescale(x):
        return x / 65_535.0

    transform = transforms.Compose([
        #transforms.ToPILImage(),
        #transforms.ToTensor(),
        transforms.Lambda(lambda x: rescale(x))
        #transforms.ConvertImageDtype(torch.float),
        #transforms.Normalize((0.5), (0.5))
    ])

    #train_dataset = DepthMapsDataset("face_embeddings_data/face_depth_maps.pt",
    #"face_embeddings_data/face_embeddings_labels.pt", transforms=transform)
    train_dataset = DepthMapsDataset(db_interface, transforms=transform)

    #print(train_dataset.depth_dims)
    #return

    train_dataloader = DataLoader(train_dataset, 8, shuffle=True)

    """ dataiter = iter(train_dataloader)
    images, labels = dataiter.next()
    print(torch.min(images), torch.max(images))
    #print(labels)
    print(images.shape) """

    model = DepthMapsClassifier(train_dataset.n_classes, train_dataset.depth_dims).to(DEVICE)

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.AdamW(model.parameters())

    n_epochs = 50

    for epoch in range(n_epochs):
        for dm, label in train_dataloader:
            dm = dm.to(DEVICE)
            label = label.to(DEVICE)

            output = model(dm)

            loss = criterion(output, label)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f'Epoch: {epoch}, Loss: {loss.item():.4f}')

    torch.save(model.state_dict(), 'depth_map_classifier_model/dm_classifier.pth')


if __name__ == "__main__":
    training()