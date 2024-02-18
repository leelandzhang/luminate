
from share import *

import pytorch_lightning as pl
# import lightning as pl
from torch.utils.data import DataLoader
#from tutorial_dataset import MyDataset

from cldm.logger import ImageLogger
from cldm.model import create_model, load_state_dict
import numpy as np

#print(os.getcwd())
#print(sys.path)
import sys
import os
from MyDataset import MyDataset

import sys
import os
sys.path.append("../rene")
print("SYS PATH",sys.path)
from rene.utils.loaders import ReneDataset

# Configs
resume_path = './models/control_sd15_ini.ckpt'
batch_size = 4
logger_freq = 300
learning_rate = 1e-5
sd_locked = True
only_mid_control = False
num_epochs = 1

# First use cpu to load models. Pytorch Lightning will automatically move it to GPUs.
model = create_model('./models/cldm_v15.yaml').cpu()
model.load_state_dict(load_state_dict(resume_path, location='cpu'))
model.learning_rate = learning_rate
model.sd_locked = sd_locked
model.only_mid_control = only_mid_control

model1 = create_model('./models/cldm_v15.yaml').cpu()
model1.load_state_dict(load_state_dict(resume_path, location='cpu'))
model1.learning_rate = learning_rate
model1.sd_locked = sd_locked
model1.only_mid_control = only_mid_control


# Misc
dataset = MyDataset()
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
logger = ImageLogger(batch_frequency=logger_freq)
optimizer = model.configure_optimizers()

for batch in dataloader:
    
    inputs = model.get_input(batch,k=0)
    print(inputs)
    1/0

for epoch in range(num_epochs):
    running_loss = 0.0
    for batch in dataloader:
        1/0
        inputs = model.get_input(batch,k=0)
        print(inputs)


        # Zero the parameter gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(inputs)
        
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if i % logger_freq == logger_freq - 1:  # Print every logger_freq mini-batches
            print(f"[{epoch + 1}, {i + 1}] loss: {running_loss / logger_freq}")
            running_loss = 0.0



# Train!
trainer.fit(model, dataloader)

