import torch.nn as nn
from roboflow import Roboflow
import base64
import numpy as np
import cv2
import torch

class lane_model(nn.Module):
  # def __init__(self):
  def __init__(self, rf_access_token):
    super(lane_model, self).__init__()
    rf = Roboflow(api_key=rf_access_token)
    project = rf.workspace().project("real_world-comp23_4")
    self.model = project.version(2).model
    # self.model = nn.Sequential(
    #   nn.Conv2d( in_channels=3 , out_channels=20 , kernel_size=15 , padding=7 , stride=1 ),
    #   nn.BatchNorm2d(20),
    #   nn.LeakyReLU(),
    #   nn.Conv2d( in_channels=20 , out_channels=10 , kernel_size=15 , padding=7 , stride=1 ),
    #   nn.BatchNorm2d(10),
    #   nn.LeakyReLU(),
    #   nn.Conv2d( in_channels=10 , out_channels=2 , kernel_size=15 , padding=7 , stride=1 ),
    # )
  def forward(self, input):
    output_json = self.model.predict(input).json()
    segmentation_mask_base64 = output_json['predictions'][0]['segmentation_mask']
    segmentation_mask_bytes = base64.b64decode(segmentation_mask_base64)
    segmentation_mask_array = np.array(bytearray(segmentation_mask_bytes), dtype=np.uint8)
    segmentation_mask = cv2.imdecode(segmentation_mask_array, cv2.IMREAD_GRAYSCALE)
    print(f"segmentation_mask.shape={segmentation_mask.shape}")
    output = segmentation_mask
    output = cv2.resize(output, (128, 128))
    output = torch.from_numpy(output)
    output_0 = torch.zeros(output.shape)
    output_0[output < .5] = 1
    output_1 = torch.zeros(output.shape)
    output_1[output >= .5] = 1
    output = torch.stack((output_0.squeeze(), output_1.squeeze()))
    output = output.unsqueeze(0)
    print(f"output.shape={output.shape}")
    # output = self.model(input)
    return output
  