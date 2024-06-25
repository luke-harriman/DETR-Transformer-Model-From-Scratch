import cv2 as cv
import numpy as np
import torch
import json
from torchvision.transforms import v2
import math
import matplotlib.pyplot as plt
from collections import Counter
from scipy.optimize import linear_sum_assignment

# Building the coco formatted dataset

class CocoDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, annotations, device, mode='train', transform=None):

      with open(root_dir + mode + annotations) as file:
        content = json.load(file)
        content['categories'][0]['name'] = 'chart'
        content['categories'][0]['supercategory'] = 'Visual'

      self.content = content
      self.root_dir = root_dir
      self.mode = mode
      self.annotations = annotations 
      self.transform = transform
      self.device = device  # Store the device

    def __len__(self):
      return len(self.content['images'])

    def __getitem__(self, idx):
        img_info = self.content['images'][idx]
        img = np.array(cv.imread(self.root_dir + self.mode + img_info['file_name']))
        H, W, C = img.shape

        if H > 1080 and W > 1920:
            # Lanczos Interpolation: Best for upsampling. Reduces alias artifacts and retains detail.
            img = cv.resize(img, (1080, 1920), interpolation=cv.INTER_LANCZOS4)
        elif H < 1080 and W < 1920:
            # Bicubic Interpolation: Considers 16 nearest pixels to estimate the new pixel values, resulting in smoother transitions and preserving more detail than bilinear interpolation.
            img = cv.resize(img, (1080, 1920), interpolation=cv.INTER_CUBIC)
        else:
            # Bilinear Interpolation: Uses a weighted average of the 4 nearest pixel values to compute the output pixel value. It’s smoother than nearest neighbor and is the default for resizing images in OpenCV.
            # For images that aren't meaningfully different from (1080, 1920). 
            img = cv.resize(img, (1080, 1920), interpolation=cv.INTER_LINEAR)
        
        # Reshape the image from (H, W, C) to (C, H, W)
        img = img.transpose(2, 0, 1)

        # Normalize the image
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        for channel, (mean_value, std_value) in enumerate(zip(mean, std)):
            img[channel, :, :] = (img[channel, :, :] - mean_value) / std_value

        # Convert to torch tensor
        img = torch.tensor(img, dtype=torch.float32).to(self.device)
        if self.transform:
            img = self.transform(img)

        # Build Labels
        target_class = torch.nn.functional.one_hot(torch.tensor(self.content['annotations'][idx]['category_id']), num_classes=2).to(self.device)

        # Ensure bbox is a list of four values
        bbox = self.content['annotations'][idx]['bbox']
        if len(bbox) != 4:
            raise ValueError("Bounding box should contain 4 values [x, y, width, height]")

        # Normalize bounding box and area
        bbox = torch.tensor(bbox, dtype=torch.float32)
        area = torch.tensor([self.content['annotations'][idx]['area']], dtype=torch.float32)
        normalize_array = torch.tensor([1080, 1920, 1080, 1920], dtype=torch.float32)
        normalized_bbox = bbox / normalize_array
        normalized_area = area / (1080 * 1920)

        # Concatenate normalized bbox and area
        target_bbox = torch.cat((normalized_bbox, normalized_area)).to(self.device) 

        return img, target_class, target_bbox



def create_linear_layers(n_embed, head_size):
    return torch.nn.ModuleDict({
        'key': torch.nn.Linear(n_embed, head_size), # Linear(input features (pixel dimensions P), output features (head_size))
        'query': torch.nn.Linear(n_embed, head_size), # Linear(input features (pixel dimensions P), output features (head_size))
        'value': torch.nn.Linear(n_embed, head_size) # Linear(input features (pixel dimensions P), output features (head_size))
    })

class AttentionBase(torch.nn.Module):
    """
    Implements the core mechanism of self-attention using linear layers.

    The linear layers have the following dimensions:
    - Input: (B, C, P)
    - Output: (B, C, head_size)

    The self-attention mechanism works by creating three different projections of the input: key, query, and value. These projections are obtained using separate linear transformations. Here's a breakdown of the process:

    1. Linear Transformation:
        - Each pixel vector in the input is linearly transformed to generate the key, query, and value vectors.
        - The dimensions of the key, query, and value vectors are controlled by the `head_size`.

    2. Computing Similarity:
        - The query vector for each pixel is compared against all key vectors using the dot product. This dot product measures the 'similarity', 'affinity', or 'likeness' between the query and the key vectors.
        - This process is similar to finding how much each pixel vector (query) 'agrees' or 'aligns' with every other pixel vector (key).

    3. Normalization:
        - The dot product values are scaled by the inverse of the square root of the `head_size` to maintain stability in gradients.
        - The scaled values are then passed through a softmax function to generate a probability distribution. This ensures that the sum of affinities for each query vector equals one, thereby making it easier to weigh the value vectors.

    4. Weighted Sum:
        - Each value vector is weighted by the corresponding softmax probability to generate the final output.
        - This process aggregates information from all pixel vectors, weighted by their similarity to the query vector.

    5. Analogy:
        - This mechanism is akin to private-public key cryptography and zk-SNARKs, where private keys (actual pixel values) are not directly shared. Instead, interactions occur using public keys (query, key vectors) to exchange information securely and efficiently.

    This class abstracts these steps, enabling the creation of attention heads used in both encoder and decoder blocks of the Transformer model.
    
    Really cool! Crazy to think something so simple is so powerful - given enough data and compute.
    """
    def __init__(self, n_embed, head_size):
        super().__init__()
        self.layers = create_linear_layers(n_embed, head_size)
        self.head_size = head_size

    def forward(self, x):
        key = self.layers['key'](x)
        query = self.layers['query'](x)
        value = self.layers['value'](x)
        weights = query @ key.transpose(-2, -1) * self.head_size ** -0.5 # (B, C, 16) @ (B, 16, C) ---> (B, C, C). The reason we need to add "* head_size**-0.5" to the weights is that without it the variance in the weights is roughly on the scale of head_size. This is important because the weights get passed into a softmax operation, which raises each value to an exponential. So if the weights are not diffused then, at initialization, the weights can take on very positive and evry negative numbers so after the softmax, the weights essentially become one-hot vectors making training extremely difficult as you get exploding or diminishing gradients.
        weights = torch.softmax(weights, dim=-1)  #  (B, C, C) @ (B, C, 16) ---> (B, C, 16)
        return weights @ value

class EncoderHead(AttentionBase):
    pass

class MultiHeadAttention(torch.nn.Module):
    def __init__(self, n_embed, n_head, head_size):
        super().__init__()
        self.heads = torch.nn.ModuleList([EncoderHead(n_embed, head_size) for _ in range(n_head)])
        self.proj = torch.nn.Linear(n_embed, n_embed)

    def forward(self, x):
        output = torch.cat([head(x) for head in self.heads], dim=-1)
        return self.proj(output)

class FeedForward(torch.nn.Module):
    def __init__(self, n_embed, projection_scale):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(n_embed, projection_scale * n_embed), # Project the input into a higher dimensional space so greater
            torch.nn.ReLU(),
            torch.nn.Linear(projection_scale * n_embed, n_embed) # Apply another Linear layer to project back to original dimension. 
        )

    def forward(self, x):
        return self.net(x)

class LayerNorm(torch.nn.Module):
    def __init__(self, dim, eps=1e-5, momentum=0.1):
        super().__init__()
        self.training = True
        self.eps = eps
        self.momentum = momentum
        self.gamma = torch.nn.Parameter(torch.ones(dim))
        self.beta = torch.nn.Parameter(torch.zeros(dim))
        self.register_buffer('running_mean', torch.zeros(dim))
        self.register_buffer('running_var', torch.ones(dim))

    def forward(self, x):
        if self.training:
            xmean = x.mean(-1, keepdim=True)
            xvar = x.var(-1, keepdim=True)
            self.running_mean = self.momentum * self.running_mean + (1 - self.momentum) * xmean
            self.running_var = self.momentum * self.running_var + (1 - self.momentum) * xvar
        else:
            xmean = self.running_mean
            xvar = self.running_var

        xhat = (x - xmean) / torch.sqrt(xvar + self.eps)
        return self.gamma * xhat + self.beta

class EncoderBlock(torch.nn.Module):
    def __init__(self, n_embed, n_aheads, head_size, projection_scale):
        super().__init__()
        self.self_attention = MultiHeadAttention(n_embed, n_aheads, head_size)
        self.feed_forward_layer = FeedForward(n_embed, projection_scale)
        self.layer_norm = LayerNorm(n_embed)

    def forward(self, x):
    # The reason we add the output from these layers to x is to create a residual connection that enables the gradient to distribute back through the network without being 'impeded' by the computation. 
    # So we are able to fork off and the computation gets added to this residual highway.
        x = x + self.self_attention(x)
        x = x + self.feed_forward_layer(x)
        return self.layer_norm(x)

class DecoderHead(AttentionBase):
    def forward(self, encoder_input, decoder_input):
        key = self.layers['key'](encoder_input)
        query = self.layers['query'](decoder_input)
        value = self.layers['value'](encoder_input)
        weights = query @ key.transpose(-2, -1) * self.head_size ** -0.5
        weights = torch.softmax(weights, dim=-1)
        return weights @ value

class EncoderDecoderHead(AttentionBase):
    def forward(self, encoder_input, decoder_input):
        key = self.layers['key'](encoder_input)
        query = self.layers['query'](decoder_input)
        value = self.layers['value'](encoder_input)
        weights = query @ key.transpose(-2, -1) * self.head_size ** -0.5
        weights = torch.softmax(weights, dim=-1)
        return weights @ value

class MultiHeadAttentionDecoder(torch.nn.Module):
    def __init__(self, n_embed, n_head, head_size, takes_encoder_input):
        super().__init__()
        self.heads = torch.nn.ModuleList(
            [EncoderDecoderHead(n_embed, head_size) if takes_encoder_input else DecoderHead(n_embed, head_size)
             for _ in range(n_head)]
        )
        self.proj = torch.nn.Linear(n_embed, n_embed)

    def forward(self, encoder_input, object_queries):
        output = torch.cat([head(encoder_input, object_queries) for head in self.heads], dim=-1)
        return self.proj(output)

class DecoderBlock(torch.nn.Module):
    def __init__(self, n_embed, n_aheads, head_size, projection_scale, takes_encoder_input):
        super().__init__()
        self.self_attention_decoder = MultiHeadAttentionDecoder(n_embed, n_aheads, head_size, takes_encoder_input=False)
        self.self_attention_encoder = MultiHeadAttentionDecoder(n_embed, n_aheads, head_size, takes_encoder_input=True)
        self.feed_forward_layer = FeedForward(n_embed, projection_scale)
        self.layer_norm = LayerNorm(n_embed)

    def forward(self, encoder_input, object_queries):
        x = self.self_attention_decoder(object_queries, object_queries) + object_queries
        x = self.layer_norm(x)
        x = self.self_attention_encoder(encoder_input, x) + x
        x = self.layer_norm(x)
        return self.feed_forward_layer(x) + x
    

def hungarian_loss(prediction, targets, batch_size, n_classes, n_predictions):
    losses = []

    for b in range(batch_size):
        target_classes = targets['class'][b].long()  # Convert to long (integer) tensor
        target_bboxes = targets['bbox'][b]

        # Ensure the number of predictions matches the number of targets
        n_targets = len(target_classes)

        # Compute cost matrix for class predictions
        class_cost_matrix = torch.zeros((n_predictions, n_targets)).to(target_classes.device)
        for x in range(n_predictions):
            for i in range(n_targets):
                class_cost_matrix[x, i] = torch.nn.functional.cross_entropy(
                    prediction['class'][b][x].unsqueeze(0),
                    target_classes[i].unsqueeze(0)
                )

        # Compute cost matrix for bounding box predictions
        bbox_cost_matrix = torch.zeros((n_predictions, n_targets)).to(target_bboxes.device)
        for x in range(n_predictions):
            for i in range(n_targets):
                bbox_cost_matrix[x, i] = torch.nn.functional.smooth_l1_loss(
                    prediction['bbox'][b][x],
                    target_bboxes[i],
                    reduction='sum'
                )

        # Combine class and bounding box cost matrices
        cost_matrix = class_cost_matrix + bbox_cost_matrix

        # Solve the linear sum assignment problem (Hungarian algorithm)
        row_ind, col_ind = linear_sum_assignment(cost_matrix.cpu().detach().numpy())

        # Compute the final loss for this batch
        batch_loss = 0
        for r, c in zip(row_ind, col_ind):
            batch_loss += class_cost_matrix[r, c] + bbox_cost_matrix[r, c]
        
        losses.append(batch_loss / len(row_ind))  # Normalize by the number of assignments

    return torch.stack(losses).mean()

class DETR(torch.nn.Module):
  def __init__(self, n_embed, n_aheads, head_size, projection_scale, d_model_embed, takes_encoder_input, N_bbox):
    super(DETR, self).__init__()
    self.backbone = CNN()
    self.conv1x1 = torch.nn.Conv2d(2048, d_model_embed, kernel_size=1) # 1x1 conv to reduce channels
    self.pe = PositionalEncoding(d_model_embed, max_len=n_embed)
    self.encoder_block = EncoderBlock(n_embed, n_aheads, head_size, projection_scale)
    self.decoder_block = DecoderBlock(n_embed, n_aheads, head_size, projection_scale, takes_encoder_input)
    self.fnn_class_mlp = torch.nn.Sequential(
      torch.nn.Linear(n_embed, 4*n_embed),
      torch.nn.ReLU(),
      torch.nn.Linear(4*n_embed, n_embed),
      torch.nn.ReLU(),
      torch.nn.Linear(n_embed, n_embed//4),
      torch.nn.ReLU(),
      torch.nn.Linear(n_embed//4, 2),
    )
    self.fnn_bbox_mlp = torch.nn.Sequential(
      torch.nn.Linear(n_embed, 4*n_embed),
      torch.nn.ReLU(),
      torch.nn.Linear(4*n_embed, n_embed),
      torch.nn.ReLU(),
      torch.nn.Linear(n_embed, n_embed//4),
      torch.nn.ReLU(),
      torch.nn.Linear(n_embed//4, N_bbox),
    )

  def forward(self, dataset, object_queries):
    x = dataset
    y = object_queries
    x = self.backbone(x)
    x = self.conv1x1(x)
    B, C, H, W = x.size()
    x = x.view(B, C, -1) # (B, 256, 1947)
    x = self.pe(x)
    x = self.encoder_block(x)
    x = self.decoder_block(x, y)
    class_logits = self.fnn_class_mlp(x)
    class_logits = torch.softmax(class_logits, dim=2)
    bbox_pred = self.fnn_bbox_mlp(x)
    return (class_logits, bbox_pred)
  

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
root_dir = '/content/drive/My Drive/Colab Notebooks/Data/output_images_coco_format/'
annotations = 'image_data_coco_format.json'


# Building Dataset
training_dataset = CocoDataset(root_dir, annotations, device=device, mode='train/')
val_dataset = CocoDataset(root_dir, annotations, device=device, mode='val/') 
test_dataset = CocoDataset(root_dir, annotations, device=device, mode='test/')

# Building Dataloaders
training_dataloader = torch.utils.data.DataLoader(training_dataset, batch_size=4, shuffle=True)
val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=4, shuffle=True)
test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=4, shuffle=True)

# Hyper Parameters
n_aheads = 11 # Number of self attention heads in each encoder/decoder block
n_embed = 1947 # Dimension size from CNN --> Encoder --> Decoder. This will remain constant as all images are resized to (1080, 1920). A future improvement can be to make this more dynamic with a probability distribution that augments the image sizes.
d_model_embed = 256 # This magic number comes from the DETR paper and represents
head_size = n_embed // n_aheads # So the dimensionality remains constant as the data moves through CNN --> Encoder --> Decoder, the output from the encoder is just the number of heads (n_aheads) multiplied by the head_size. This output must equal the dimensionality from the CNN (n_embed).
projection_scale = 4  # Multiple at which the input is scaled in the MLP; represents the input in higher dimensional space in the feedford layer of the encoder block.
N = 100
takes_encoder_input = True
N_bbox = 5
optimizer_lr = 1e-4
epochs = 20

# Detr Model and Arbitrary Object Queries
detr_model = DETR(n_embed, n_aheads, head_size, projection_scale, d_model_embed, takes_encoder_input, N_bbox).to(device)
object_queries = torch.randn(N, n_embed, requires_grad=True).to(device)

# Loss
class_loss = torch.nn.CrossEntropyLoss()
bbox_loss = torch.nn.SmoothL1Loss()

# Optimizers
optimizer = torch.optim.Adam(detr_model.parameters(), lr=optimizer_lr)


# Training loop
data = [] # Data for matplotlib
for epoch in range(epochs):
    detr_model.train()
    running_loss = 0.0

    for batch_data, batch_classes, batch_bboxes in training_dataloader:
        batch_data = batch_data.to(device)
        batch_classes = batch_classes.to(device)
        batch_bboxes = batch_bboxes.to(device)
        optimizer.zero_grad()

        # Prepare targets for Hungarian loss
        targets = {
            'class': batch_classes,
            'bbox': batch_bboxes
        }

        # Forward pass
        class_logits, bbox_pred = detr_model(batch_data, object_queries)

        # Prepare predictions for Hungarian loss
        prediction = {
            'class': class_logits,
            'bbox': bbox_pred
        }
        # Compute Hungarian loss
        loss = hungarian_loss(prediction, targets, batch_size=batch_data.size(0), n_classes=2, n_predictions=N)

        # Backward pass and optimization
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        data.append([epoch, running_loss, loss.item()])
        print(f'Running Loss: {running_loss}, Loss: {loss.item()}')
    
    # Save the model checkpoint at the end of each epoch
    torch.save({
        'epoch': epoch,
        'model_state_dict': detr_model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }, f'detr_model_epoch_{epoch}_loss_{data[-1]}.pth')

print("Training complete.")