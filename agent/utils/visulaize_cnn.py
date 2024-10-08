import torch
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image
import matplotlib.pyplot as plt
from utils.path_generator import PathGenerator

# Load the pre-trained ResNet18 model
resnet18 = models.resnet18(pretrained=True)
resnet18.eval()  # Set the model to evaluation mode

pg = PathGenerator(12.0, 14.0, [-25, 25], [0, 22], 2)
goal_trajectory = pg.generate_path()
image_np = pg.generate_floor_plan(goal_trajectory)
image_pil = Image.fromarray(image_np.squeeze()).convert('RGB')

preprocess = transforms.Compose([
    transforms.
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
input_tensor = preprocess(image_pil)
input_batch = input_tensor.unsqueeze(0)  # Add batch dimension

# Forward pass through the model
with torch.no_grad():
    activations = resnet18(input_batch)

# Visualize the activations of a specific layer
target_layer_index = 4  # Choose the index of the layer you want to visualize
activations = activations[:, target_layer_index]

# Plot the activations as images
num_filters = activations.size(0)
fig, axes = plt.subplots(1, num_filters, figsize=(20, 5))
for i in range(num_filters):
    activation_map = activations[i, :, :]
    axes[i].imshow(activation_map, cmap='viridis')
    axes[i].axis('off')
    axes[i].set_title(f'Activation Map {i+1}')
plt.show()
print()