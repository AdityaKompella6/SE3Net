from point_cloud_helpers import convert_depth_image_to_point_cloud,resize_point_cloud
import json
import numpy as np
from PIL import Image
import torch
from se3net import SE3Net
import matplotlib.pyplot as plt
num_epochs = 1000
model = SE3Net(6, 5).to("cuda")
optimizer = torch.optim.Adam(model.parameters(),5e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,num_epochs,5e-6)
loss_fn = torch.nn.MSELoss()
action = torch.randn((1,5)).to("cuda")
##Inputs
depth_img = np.array(Image.open("0000_depth.png"))
f = open("hope_video_scene_0009/0000.json")
data = json.load(f)
extrinsic_matrix = np.array(data["camera"]['extrinsics'])
intrinsic_matrix = np.array(data["camera"]['intrinsics'])
inputs = convert_depth_image_to_point_cloud(depth_img,intrinsic_matrix,extrinsic_matrix)
inputs = resize_point_cloud(inputs)
##Outputs
depth_img = np.array(Image.open("0009_depth.png"))
f = open("hope_video_scene_0009/0009.json")
data = json.load(f)
extrinsic_matrix = np.array(data["camera"]['extrinsics'])
intrinsic_matrix = np.array(data["camera"]['intrinsics'])
outputs = convert_depth_image_to_point_cloud(depth_img,intrinsic_matrix,extrinsic_matrix)
outputs = resize_point_cloud(outputs)
channel_mean = torch.mean(outputs,dim=(2,3),keepdim=True)
channel_std = torch.std(outputs,dim=(2,3),keepdim=True)
##Standardize inputs and outputs
inputs = (inputs-channel_std)/channel_mean
outputs = (outputs-channel_std)/channel_mean
inputs = inputs.to("cuda")
outputs = outputs.to("cuda")
for i in range(num_epochs):
    pred = model(inputs,action)
    loss = loss_fn(pred,outputs)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    scheduler.step()
    if i % 100 == 0:
        print(loss.item())
        plt.imshow(pred[0,2,:,:].detach().cpu().numpy())
        plt.show()
