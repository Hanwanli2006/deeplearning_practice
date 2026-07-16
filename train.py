  import torch
  import torch.nn as nn
  import torch.optim as optim
  from torch.utils.data import Dataset, DataLoader
  import os
  import cv2
  import numpy as np
  from tqdm import tqdm
  import random

  torch.manual_seed(42)
  np.random.seed(42)
  random.seed(42)

  # ========== 配置 ==========
  DATA_DIR = "/kaggle/input/datasets/matthewjansen/ucf101-action-recognition"
  SAVE_DIR = "/kaggle/working/preprocessed"
  BATCH_SIZE = 8
  EPOCHS = 50
  LR = 1e-4
  FRAME_LEN = 16
  FRAME_SIZE = 112



  print("开始预处理视频...")
  for split in ["train", "val"]:
      split_dir = os.path.join(DATA_DIR, split)
      for cls in tqdm(os.listdir(split_dir), desc=f"预处理 {split}"):
          cls_dir = os.path.join(split_dir, cls)
          if not os.path.isdir(cls_dir):
              continue
          out_dir = os.path.join(SAVE_DIR, split, cls)
          os.makedirs(out_dir, exist_ok=True)
          for vid in os.listdir(cls_dir):
              if not vid.endswith(".avi"):
                  continue
              out_path = os.path.join(out_dir, vid.replace(".avi", ".npy"))
              if os.path.exists(out_path):
                  continue
              cap = cv2.VideoCapture(os.path.join(cls_dir, vid))
              total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
              indices = np.linspace(0, total-1, FRAME_LEN, dtype=int)
              frames = []
              for i in indices:
                  cap.set(cv2.CAP_PROP_POS_FRAMES, i)
                  ret, frame = cap.read()
                  if not ret:
                      frame = frames[-1] if frames else np.zeros((FRAME_SIZE, FRAME_SIZE, 3), dtype=np.uint8)
                  frame = cv2.resize(frame, (FRAME_SIZE, FRAME_SIZE))
                  frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                  frames.append(frame)
              cap.release()
              np.save(out_path, np.stack(frames, axis=0).astype(np.uint8))
  print("预处理完成！\n")



  class VideoDataset(Dataset):
      def __init__(self, root_dir, split="train"):
          self.samples = []
          split_dir = os.path.join(root_dir, split)
          classes = sorted(os.listdir(split_dir))[:5]
          self.class_to_idx = {c: i for i, c in enumerate(classes)}
          for cls in classes:
              cls_dir = os.path.join(split_dir, cls)
              if not os.path.isdir(cls_dir):
                  continue
              for f in os.listdir(cls_dir):
                  if f.endswith(".npy"):
                      self.samples.append((os.path.join(cls_dir, f), self.class_to_idx[cls]))

      def __len__(self):
          return len(self.samples)

      def __getitem__(self, idx):
          path, label = self.samples[idx]
          video = np.load(path).astype(np.float32) / 255.0
          video = torch.from_numpy(video).permute(3, 0, 1, 2)
          return (video - 0.5) / 0.5, label

class C3D(nn.Module):
      def __init__(self, num_classes=101):
          super().__init__()
          self.conv1 = nn.Conv3d(3, 64, kernel_size=3, padding=1)
          self.pool1 = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))

          self.conv2 = nn.Conv3d(64, 128, kernel_size=3, padding=1)
          self.pool2 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))

          self.conv3a = nn.Conv3d(128, 256, kernel_size=3, padding=1)
          self.conv3b = nn.Conv3d(256, 256, kernel_size=3, padding=1)
          self.pool3 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))

          self.conv4a = nn.Conv3d(256, 512, kernel_size=3, padding=1)
          self.conv4b = nn.Conv3d(512, 512, kernel_size=3, padding=1)
          self.pool4 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))

          self.conv5a = nn.Conv3d(512, 512, kernel_size=3, padding=1)
          self.conv5b = nn.Conv3d(512, 512, kernel_size=3, padding=1)
          self.pool5 = nn.AdaptiveAvgPool3d((1, 2, 2))

          self.fc6 = nn.Linear(512 * 1 * 2 * 2, 4096)
          self.fc7 = nn.Linear(4096, 4096)
          self.fc8 = nn.Linear(4096, num_classes)
          self.dropout = nn.Dropout(0.5)
          self.relu = nn.ReLU()

      def forward(self, x):
          x = self.relu(self.conv1(x))
          x = self.pool1(x)

          x = self.relu(self.conv2(x))
          x = self.pool2(x)

          x = self.relu(self.conv3a(x))
          x = self.relu(self.conv3b(x))
          x = self.pool3(x)

          x = self.relu(self.conv4a(x))
          x = self.relu(self.conv4b(x))
          x = self.pool4(x)

          x = self.relu(self.conv5a(x))
          x = self.relu(self.conv5b(x))
          x = self.pool5(x)

          x = x.view(x.size(0), -1)
          x = self.relu(self.fc6(x))
          x = self.dropout(x)
          x = self.relu(self.fc7(x))
          x = self.dropout(x)
          x = self.fc8(x)
          return x
      
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

  train_dataset = VideoDataset(SAVE_DIR, split="train")
  val_dataset = VideoDataset(SAVE_DIR, split="val")
  NUM_CLASSES = len(train_dataset.class_to_idx)
  print(f"类别数: {NUM_CLASSES}, 训练样本: {len(train_dataset)}, 验证样本: {len(val_dataset)}")

  train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)
  val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)

  model = C3D(num_classes=NUM_CLASSES).to(device)
  criterion = nn.CrossEntropyLoss()
  optimizer = optim.Adam(model.parameters(), lr=LR)
  scaler = torch.cuda.amp.GradScaler()

  for epoch in range(EPOCHS):
      model.train()
      correct = 0
      total = 0
      loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")
      for videos, labels in loop:
          videos, labels = videos.to(device), labels.to(device)
          with torch.cuda.amp.autocast():
              outputs = model(videos)
              loss = criterion(outputs, labels)
          optimizer.zero_grad()
          scaler.scale(loss).backward()
          scaler.step(optimizer)
          scaler.update()
          _, predicted = outputs.max(1)
          total += labels.size(0)
          correct += predicted.eq(labels).sum().item()
          loop.set_postfix(acc=f"{100.*correct/total:.1f}%")

      # 验证
      model.eval()
      val_correct = 0
      val_total = 0
      with torch.no_grad():
          for videos, labels in val_loader:
              videos, labels = videos.to(device), labels.to(device)
              outputs = model(videos)
              _, predicted = outputs.max(1)
              val_total += labels.size(0)
              val_correct += predicted.eq(labels).sum().item()

      print(f"Epoch {epoch+1}: Train Acc {100.*correct/total:.2f}% | Val Acc {100.*val_correct/val_total:.2f}%")
