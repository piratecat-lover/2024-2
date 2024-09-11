import time
import math
import torch 
import torch.nn as nn
from torch import optim
import torchvision
from a5_helper import *
import matplotlib.pyplot as plt


def hello_single_stage_detector():
    print("Hello from single_stage_detector.py!")


def GenerateAnchor(anc, grid):
  """
  Anchor generator.

  Inputs:
  - anc: Tensor of shape (A, 2) giving the shapes of anchor boxes to consider at
    each point in the grid. anc[a] = (w, h) gives the width and height of the
    a'th anchor shape.
  - grid: Tensor of shape (B, H', W', 2) giving the (x, y) coordinates of the
    center of each feature from the backbone feature map. This is the tensor
    returned from GenerateGrid.
  
  Outputs:
  - anchors: Tensor of shape (B, A, H', W', 4) giving the positions of all
    anchor boxes for the entire image. anchors[b, a, h, w] is an anchor box
    centered at grid[b, h, w], whose shape is given by anc[a]; we parameterize
    boxes as anchors[b, a, h, w] = (x_tl, y_tl, x_br, y_br), where (x_tl, y_tl)
    and (x_br, y_br) give the xy coordinates of the top-left and bottom-right
    corners of the box.
  """
  anchors = None
  ##############################################################################
  # TODO: Given a set of anchor shapes and a grid cell on the activation map,  #
  # generate all the anchor coordinates for each image. Support batch input.   #
  ##############################################################################
  # Replace "pass" statement with your code
  B, H, W, _ = grid.shape
  A = anc.shape[0]

  anchors = torch.zeros(B, A, H, W, 4, device=grid.device, dtype=grid.dtype)

  for a in range(A):
      w_a, h_a = anc[a]
      anchors[:, a, :, :, 0] = grid[:, :, :, 0] - w_a / 2.0  # x_tl
      anchors[:, a, :, :, 1] = grid[:, :, :, 1] - h_a / 2.0  # y_tl
      anchors[:, a, :, :, 2] = grid[:, :, :, 0] + w_a / 2.0  # x_br
      anchors[:, a, :, :, 3] = grid[:, :, :, 1] + h_a / 2.0  # y_br
  ##############################################################################
  #                               END OF YOUR CODE                             #
  ##############################################################################

  return anchors


def GenerateProposal(anchors, offsets, method='YOLO'):
  """
  Proposal generator.

  Inputs:
  - anchors: Anchor boxes, of shape (B, A, H', W', 4). Anchors are represented
    by the coordinates of their top-left and bottom-right corners.
  - offsets: Transformations of shape (B, A, H', W', 4) that will be used to
    convert anchor boxes into region proposals. The transformation
    offsets[b, a, h, w] = (tx, ty, tw, th) will be applied to the anchor
    anchors[b, a, h, w]. For YOLO, assume that tx and ty are in the range
    (-0.5, 0.5).
  - method: Which transformation formula to use, either 'YOLO' or 'FasterRCNN'
  
  Outputs:
  - proposals: Region proposals of shape (B, A, H', W', 4), represented by the
    coordinates of their top-left and bottom-right corners. Applying the
    transform offsets[b, a, h, w] to the anchor [b, a, h, w] should give the
    proposal proposals[b, a, h, w].
  
  """
  assert(method in ['YOLO', 'FasterRCNN'])
  proposals = None
  ##############################################################################
  # TODO: Given anchor coordinates and the proposed offset for each anchor,    #
  # compute the proposal coordinates using the transformation formulas above.  #
  ##############################################################################
  # Replace "pass" statement with your code
  B, A, H, W, _ = anchors.shape
  proposals = torch.zeros_like(anchors)

  if method == 'YOLO':
      tx = offsets[..., 0].sigmoid() - 0.5
      ty = offsets[..., 1].sigmoid() - 0.5
      tw = offsets[..., 2]
      th = offsets[..., 3]

      proposals[..., 0] = anchors[..., 0] + tx * (anchors[..., 2] - anchors[..., 0])
      proposals[..., 1] = anchors[..., 1] + ty * (anchors[..., 3] - anchors[..., 1])
      proposals[..., 2] = (anchors[..., 2] - anchors[..., 0]) * tw.exp() + anchors[..., 0]
      proposals[..., 3] = (anchors[..., 3] - anchors[..., 1]) * th.exp() + anchors[..., 1]
  ##############################################################################
  #                               END OF YOUR CODE                             #
  ##############################################################################

  return proposals


def IoU(proposals, bboxes):
  """
  Compute intersection over union between sets of bounding boxes.

  Inputs:
  - proposals: Proposals of shape (B, A, H', W', 4)
  - bboxes: Ground-truth boxes from the DataLoader of shape (B, N, 5).
    Each ground-truth box is represented as tuple (x_lr, y_lr, x_rb, y_rb, class).
    If image i has fewer than N boxes, then bboxes[i] will be padded with extra
    rows of -1.
  
  Outputs:
  - iou_mat: IoU matrix of shape (B, A*H'*W', N) where iou_mat[b, i, n] gives
    the IoU between one element of proposals[b] and bboxes[b, n].

  For this implementation you DO NOT need to filter invalid proposals or boxes;
  in particular you don't need any special handling for bboxxes that are padded
  with -1.
  """
  iou_mat = None
  ##############################################################################
  # TODO: Compute the Intersection over Union (IoU) on proposals and GT boxes. #
  # No need to filter invalid proposals/bboxes (i.e., allow region area <= 0). #
  # However, you need to make sure to compute the IoU correctly (it should be  #
  # 0 in those cases.                                                          # 
  # You need to ensure your implementation is efficient (no for loops).        #
  # HINT:                                                                      #
  # IoU = Area of Intersection / Area of Union, where                          #
  # Area of Union = Area of Proposal + Area of BBox - Area of Intersection     #
  # and the Area of Intersection can be computed using the top-left corner and #
  # bottom-right corner of proposal and bbox. Think about their relationships. #
  ##############################################################################
  # Replace "pass" statement with your code
  B, A, H, W, _ = proposals.shape  # (B, A, H', W', 4)
  N = bboxes.shape[1]  # Number of ground-truth boxes

  # Reshape proposals to (B, A*H*W, 4) for easier processing
  proposals = proposals.view(B, A * H * W, 4)

  # Extract the top-left and bottom-right coordinates of proposals and bboxes
  proposal_x1, proposal_y1 = proposals[..., 0], proposals[..., 1]  # top-left (x, y)
  proposal_x2, proposal_y2 = proposals[..., 2], proposals[..., 3]  # bottom-right (x, y)
  bbox_x1, bbox_y1 = bboxes[..., 0], bboxes[..., 1]  # top-left (x, y)
  bbox_x2, bbox_y2 = bboxes[..., 2], bboxes[..., 3]  # bottom-right (x, y)

  # Calculate the coordinates of the intersection rectangle
  inter_x1 = torch.max(proposal_x1.unsqueeze(-1), bbox_x1.unsqueeze(1))  # (B, A*H*W, N)
  inter_y1 = torch.max(proposal_y1.unsqueeze(-1), bbox_y1.unsqueeze(1))  # (B, A*H*W, N)
  inter_x2 = torch.min(proposal_x2.unsqueeze(-1), bbox_x2.unsqueeze(1))  # (B, A*H*W, N)
  inter_y2 = torch.min(proposal_y2.unsqueeze(-1), bbox_y2.unsqueeze(1))  # (B, A*H*W, N)

  # Compute intersection width and height
  inter_w = (inter_x2 - inter_x1).clamp(min=0)  # Clamp to avoid negative values
  inter_h = (inter_y2 - inter_y1).clamp(min=0)  # Clamp to avoid negative values

  # Compute the area of the intersection
  inter_area = inter_w * inter_h  # (B, A*H*W, N)

  # Compute the area of the proposals and ground-truth boxes
  proposal_area = (proposal_x2 - proposal_x1) * (proposal_y2 - proposal_y1)  # (B, A*H*W)
  bbox_area = (bbox_x2 - bbox_x1) * (bbox_y2 - bbox_y1)  # (B, N)

  # Compute the area of the union
  union_area = proposal_area.unsqueeze(-1) + bbox_area.unsqueeze(1) - inter_area  # (B, A*H*W, N)

  # Compute IoU as the ratio of intersection area to union area
  iou_mat = inter_area / union_area  # (B, A*H*W, N)

  ##############################################################################
  #                               END OF YOUR CODE                             #
  ##############################################################################
  return iou_mat


class PredictionNetwork(nn.Module):
  def __init__(self, in_dim, hidden_dim=128, num_anchors=9, num_classes=20, drop_ratio=0.3):
    super().__init__()

    assert(num_classes != 0 and num_anchors != 0)
    self.num_classes = num_classes
    self.num_anchors = num_anchors

    ##############################################################################
    # TODO: Set up a network that will predict outputs for all anchors. This     #
    # network should have a 1x1 convolution with hidden_dim filters, followed    #
    # by a Dropout layer with p=drop_ratio, a Leaky ReLU nonlinearity, and       #
    # finally another 1x1 convolution layer to predict all outputs. You can      #
    # use an nn.Sequential for this network, and store it in a member variable.  #
    # HINT: The output should be of shape (B, 5*A+C, 7, 7), where                #
    # A=self.num_anchors and C=self.num_classes.                                 #
    ##############################################################################
    # Make sure to name your prediction network pred_layer.
    self.pred_layer = None
    # Replace "pass" statement with your code
    self.pred_layer = nn.Sequential(
            nn.Conv2d(in_dim, hidden_dim, kernel_size=1),  # 1x1 convolution
            nn.Dropout(p=drop_ratio),                      # Dropout layer
            nn.LeakyReLU(negative_slope=0.1),              # Leaky ReLU activation
            nn.Conv2d(hidden_dim, num_anchors * (5 + num_classes), kernel_size=1)  # Final 1x1 conv layer
        )
    ##############################################################################
    #                               END OF YOUR CODE                             #
    ##############################################################################

  def _extract_anchor_data(self, anchor_data, anchor_idx):
    """
    Inputs:
    - anchor_data: Tensor of shape (B, A, D, H, W) giving a vector of length
      D for each of A anchors at each point in an H x W grid.
    - anchor_idx: int64 Tensor of shape (M,) giving anchor indices to extract

    Returns:
    - extracted_anchors: Tensor of shape (M, D) giving anchor data for each
      of the anchors specified by anchor_idx.
    """
    B, A, D, H, W = anchor_data.shape
    anchor_data = anchor_data.permute(0, 1, 3, 4, 2).contiguous().view(-1, D)
    extracted_anchors = anchor_data[anchor_idx]
    return extracted_anchors
  
  def _extract_class_scores(self, all_scores, anchor_idx):
    """
    Inputs:
    - all_scores: Tensor of shape (B, C, H, W) giving classification scores for
      C classes at each point in an H x W grid.
    - anchor_idx: int64 Tensor of shape (M,) giving the indices of anchors at
      which to extract classification scores

    Returns:
    - extracted_scores: Tensor of shape (M, C) giving the classification scores
      for each of the anchors specified by anchor_idx.
    """
    B, C, H, W = all_scores.shape
    A = self.num_anchors
    all_scores = all_scores.contiguous().permute(0, 2, 3, 1).contiguous()
    all_scores = all_scores.view(B, 1, H, W, C).expand(B, A, H, W, C)
    all_scores = all_scores.reshape(B * A * H * W, C)
    extracted_scores = all_scores[anchor_idx]
    return extracted_scores

  def forward(self, features, pos_anchor_idx=None, neg_anchor_idx=None):
    """
    Run the forward pass of the network to predict outputs given features
    from the backbone network.

    Inputs:
    - features: Tensor of shape (B, in_dim, 7, 7) giving image features computed
      by the backbone network.
    - pos_anchor_idx: int64 Tensor of shape (M,) giving the indices of anchors
      marked as positive. These are only given during training; at test-time
      this should be None.
    - neg_anchor_idx: int64 Tensor of shape (M,) giving the indices of anchors
      marked as negative. These are only given at training; at test-time this
      should be None.
    
    The outputs from this method are different during training and inference.
    
    During training, pos_anchor_idx and neg_anchor_idx are given and identify
    which anchors should be positive and negative, and this forward pass needs
    to extract only the predictions for the positive and negative anchors.

    During inference, only features are provided and this method needs to return
    predictions for all anchors.

    Outputs (During training):
    - conf_scores: Tensor of shape (2*M, 1) giving the predicted classification
      scores for positive anchors and negative anchors (in that order).
    - offsets: Tensor of shape (M, 4) giving predicted transformation for
      positive anchors.
    - class_scores: Tensor of shape (M, C) giving classification scores for
      positive anchors.

    Outputs (During inference):
    - conf_scores: Tensor of shape (B, A, H, W) giving predicted classification
      scores for all anchors.
    - offsets: Tensor of shape (B, A, 4, H, W) giving predicted transformations
      all all anchors.
    - class_scores: Tensor of shape (B, C, H, W) giving classification scores for
      each spatial position.
    """
    conf_scores, offsets, class_scores = None, None, None
    ############################################################################
    # TODO: Use backbone features to predict conf_scores, offsets, and         #
    # class_scores. Make sure conf_scores is between 0 and 1 by squashing the  #
    # network output with a sigmoid. Also make sure the first two elements t^x #
    # and t^y of offsets are between -0.5 and 0.5 by squashing with a sigmoid  #
    # and subtracting 0.5.                                                     #
    #                                                                          #
    # During training you need to extract the outputs for only the positive    #
    # and negative anchors as specified above.                                 #
    #                                                                          #
    # HINT: You can use the provided helper methods self._extract_anchor_data  #
    # and self._extract_class_scores to extract information for positive and   #
    # negative anchors specified by pos_anchor_idx and neg_anchor_idx.         #
    ############################################################################
    # Replace "pass" statement with your code
    predictions = self.pred_layer(features)
    B, A_5C, H, W = predictions.shape
    A = self.num_anchors
    C = self.num_classes
    
    # Split the predictions into confidence scores, offsets, and class scores
    predictions = predictions.view(B, A, 5 + C, H, W)
    conf_scores_raw = predictions[:, :, 4, :, :]  # Shape (B, A, H, W)
    offsets_raw = predictions[:, :, :4, :, :]     # Shape (B, A, 4, H, W)
    class_scores_raw = predictions[:, :, 5:, :, :]  # Shape (B, A, C, H, W)
    
    # Process confidence scores with sigmoid to bring values between 0 and 1
    conf_scores = torch.sigmoid(conf_scores_raw)  # Shape (B, A, H, W)
    
    # Process offsets, ensuring t^x and t^y are between -0.5 and 0.5
    offsets = torch.clone(offsets_raw)
    offsets[:, :, 0:2, :, :] = torch.sigmoid(offsets[:, :, 0:2, :, :]) - 0.5
    
    # During training, extract only the positive and negative anchors
    if pos_anchor_idx is not None and neg_anchor_idx is not None:
        # Extract the confidence scores for positive and negative anchors
        pos_conf_scores = self._extract_anchor_data(conf_scores.unsqueeze(2), pos_anchor_idx)
        neg_conf_scores = self._extract_anchor_data(conf_scores.unsqueeze(2), neg_anchor_idx)
        conf_scores = torch.cat([pos_conf_scores, neg_conf_scores], dim=0)  # Shape (2*M, 1)

        # Extract the offsets for positive anchors only
        offsets = self._extract_anchor_data(offsets, pos_anchor_idx)  # Shape (M, 4)

        # Extract the class scores for positive anchors only
        class_scores = self._extract_class_scores(class_scores_raw, pos_anchor_idx)  # Shape (M, C)

    # During inference, return outputs for all anchors
    else:
        class_scores = class_scores_raw.view(B, C, H, W)  # Shape (B, C, H, W)

    return conf_scores, offsets, class_scores
    ##############################################################################
    #                               END OF YOUR CODE                             #
    ##############################################################################
    return conf_scores, offsets, class_scores


class SingleStageDetector(nn.Module):
  def __init__(self):
    super().__init__()

    self.anchor_list = torch.tensor([[1., 1], [2, 2], [3, 3], [4, 4], [5, 5], [2, 3], [3, 2], [3, 5], [5, 3]]) # READ ONLY
    self.feat_extractor = FeatureExtractor()
    self.num_classes = 20
    self.pred_network = PredictionNetwork(1280, num_anchors=self.anchor_list.shape[0], \
                                          num_classes=self.num_classes)
  def forward(self, images, bboxes):
    """
    Training-time forward pass for the single-stage detector.

    Inputs:
    - images: Input images, of shape (B, 3, 224, 224)
    - bboxes: GT bounding boxes of shape (B, N, 5) (padded)

    Outputs:
    - total_loss: Torch scalar giving the total loss for the batch.
    """
    # weights to multiple to each loss term
    w_conf = 1 # for conf_scores
    w_reg = 1 # for offsets
    w_cls = 1 # for class_prob

    total_loss = None
    ##############################################################################
    # TODO: Implement the forward pass of SingleStageDetector.                   #
    # A few key steps are outlined as follows:                                   #
    # i) Image feature extraction,                                               #
    # ii) Grid and anchor generation,                                            #
    # iii) Compute IoU between anchors and GT boxes and then determine activated/#
    #      negative anchors, and GT_conf_scores, GT_offsets, GT_class,           #
    # iv) Compute conf_scores, offsets, class_prob through the prediction network#
    # v) Compute the total_loss which is formulated as:                          #
    #    total_loss = w_conf * conf_loss + w_reg * reg_loss + w_cls * cls_loss,  #
    #    where conf_loss is determined by ConfScoreRegression, w_reg by          #
    #    BboxRegression, and w_cls by ObjectClassification.                      #
    # HINT: Set `neg_thresh=0.2` in ReferenceOnActivatedAnchors in this notebook #
    #       (A5-1) for a better performance than with the default value.         #
    ##############################################################################
    # Replace "pass" statement with your code
    features = self.feat_extractor(images)  # Extract features (B, 1280, 7, 7)
    
    B, _, H, W = features.shape
    grid = GenerateGrid(B, W, H, device=images.device)  # Generate the grid
    anchors = GenerateAnchor(self.anchor_list.to(images.device), grid)  # Generate anchors
    
    iou_mat = IoU(anchors, bboxes)  # Shape (B, A*H*W, N)
    pos_anchor_idx, neg_anchor_idx, GT_conf_scores, GT_offsets, GT_class = ReferenceOnActivatedAnchors(anchors, bboxes, grid, iou_mat, neg_thresh=0.2)
        
    conf_scores, offsets, class_scores = self.pred_network(features, pos_anchor_idx=pos_anchor_idx, neg_anchor_idx=neg_anchor_idx)
    
    conf_loss = ConfScoreRegression(conf_scores, GT_conf_scores)
    reg_loss = BboxRegression(offsets, GT_offsets)
    cls_loss = ObjectClassification(class_scores, GT_class, self.num_classes)

    total_loss = w_conf * conf_loss + w_reg * reg_loss + w_cls * cls_loss

    ##############################################################################
    #                               END OF YOUR CODE                             #
    ##############################################################################

    return total_loss
  
  def inference(self, images, thresh=0.5, nms_thresh=0.7):
    """"
    Inference-time forward pass for the single stage detector.

    Inputs:
    - images: Input images
    - thresh: Threshold value on confidence scores
    - nms_thresh: Threshold value on NMS

    Outputs:
    - final_propsals: Keeped proposals after confidence score thresholding and NMS,
                      a list of B (*x4) tensors
    - final_conf_scores: Corresponding confidence scores, a list of B (*x1) tensors
    - final_class: Corresponding class predictions, a list of B  (*x1) tensors
    """
    final_proposals, final_conf_scores, final_class = [], [], []
    ##############################################################################
    # TODO: Predicting the final proposal coordinates `final_proposals`,         #
    # confidence scores `final_conf_scores`, and the class index `final_class`.  #
    # The overall steps are similar to the forward pass but now you do not need  #
    # to decide the activated nor negative anchors.                              #
    # HINT: Thresholding the conf_scores based on the threshold value `thresh`.  #
    # Then, apply NMS (torchvision.ops.nms) to the filtered proposals given the  #
    # threshold `nms_thresh`.                                                    #
    # The class index is determined by the class with the maximal probability.   #
    # Note that `final_propsals`, `final_conf_scores`, and `final_class` are all #
    # lists of B 2-D tensors (you may need to unsqueeze dim=1 for the last two). #
    ##############################################################################
    # Replace "pass" statement with your code
    features = self.feat_extractor(images)  # Extract features (B, 1280, 7, 7)
    
    B, _, H, W = features.shape
    grid = GenerateGrid(B, W, H, device=images.device)  # Generate the grid
    anchors = GenerateAnchor(self.anchor_list.to(images.device), grid)  # Generate anchors
    
    conf_scores, offsets, class_scores = self.pred_network(features)  # Predict for all anchors
    
    for i in range(B):
        # Filter out proposals with confidence score greater than the threshold
        anchor_i = anchors[i].view(-1, 4)  # Shape (A*H*W, 4)
        conf_i = conf_scores[i].view(-1)  # Shape (A*H*W,)
        offsets_i = offsets[i].view(-1, 4)  # Shape (A*H*W, 4)
        class_scores_i = class_scores[i].view(-1, self.num_classes)  # Shape (A*H*W, C)

        # Apply threshold on confidence scores
        keep_idx = conf_i > thresh
        anchor_i = anchor_i[keep_idx]
        conf_i = conf_i[keep_idx].unsqueeze(1)
        offsets_i = offsets_i[keep_idx]
        class_scores_i = class_scores_i[keep_idx]

        # Apply offsets to get final proposals (manually apply the transformations)
        proposals = self.apply_offsets_to_anchors(anchor_i, offsets_i) # Requires new method

        # Apply NMS to remove overlapping boxes
        nms_idx = torchvision.ops.nms(proposals, conf_i.squeeze(), nms_thresh)
        final_proposals.append(proposals[nms_idx])
        final_conf_scores.append(conf_i[nms_idx].unsqueeze(1))
        final_class.append(class_scores_i[nms_idx].argmax(dim=1, keepdim=True))

    return final_proposals, final_conf_scores, final_class

def apply_offsets_to_anchors(self, anchors, offsets):
    """
    NEW METHOD ADDITION BY CHATGPT
    Applies the predicted offsets to the anchors to obtain the final proposals.

    Inputs:
    - anchors: Tensor of shape (M, 4) giving the coordinates of the anchors (x1, y1, x2, y2)
    - offsets: Tensor of shape (M, 4) giving the predicted offsets (tx, ty, tw, th)

    Outputs:
    - proposals: Tensor of shape (M, 4) giving the final bounding boxes (x1, y1, x2, y2)
    """
    # Compute width, height, and center (cx, cy) for the anchors
    anchor_widths = anchors[:, 2] - anchors[:, 0]
    anchor_heights = anchors[:, 3] - anchors[:, 1]
    anchor_cx = anchors[:, 0] + 0.5 * anchor_widths
    anchor_cy = anchors[:, 1] + 0.5 * anchor_heights

    # Extract the offsets (tx, ty, tw, th)
    tx = offsets[:, 0]
    ty = offsets[:, 1]
    tw = offsets[:, 2]
    th = offsets[:, 3]

    # Apply the transformations
    pred_cx = anchor_cx + tx * anchor_widths
    pred_cy = anchor_cy + ty * anchor_heights
    pred_w = anchor_widths * torch.exp(tw)
    pred_h = anchor_heights * torch.exp(th)

    # Convert center (cx, cy), width (w), height (h) to (x1, y1, x2, y2)
    pred_x1 = pred_cx - 0.5 * pred_w
    pred_y1 = pred_cy - 0.5 * pred_h
    pred_x2 = pred_cx + 0.5 * pred_w
    pred_y2 = pred_cy + 0.5 * pred_h

    # Stack to form final proposals
    proposals = torch.stack([pred_x1, pred_y1, pred_x2, pred_y2], dim=1)
    return proposals


def nms(boxes, scores, iou_threshold=0.5, topk=None):
  """
  Non-maximum suppression removes overlapping bounding boxes.

  Inputs:
  - boxes: top-left and bottom-right coordinate values of the bounding boxes
    to perform NMS on, of shape Nx4
  - scores: scores for each one of the boxes, of shape N
  - iou_threshold: discards all overlapping boxes with IoU > iou_threshold; float
  - topk: If this is not None, then return only the topk highest-scoring boxes.
    Otherwise if this is None, then return all boxes that pass NMS.

  Outputs:
  - keep: torch.long tensor with the indices of the elements that have been
    kept by NMS, sorted in decreasing order of scores; of shape [num_kept_boxes]
  """

  if (not boxes.numel()) or (not scores.numel()):
    return torch.zeros(0, dtype=torch.long)

  keep = None
  #############################################################################
  # TODO: Implement non-maximum suppression which iterates the following:     #
  #       1. Select the highest-scoring box among the remaining ones,         #
  #          which has not been chosen in this step before                    #
  #       2. Eliminate boxes with IoU > threshold                             #
  #       3. If any boxes remain, GOTO 1                                      #
  #       Your implementation should not depend on a specific device type;    #
  #       you can use the device of the input if necessary.                   #
  # HINT: You can refer to the torchvision library code:                      #
  #   github.com/pytorch/vision/blob/master/torchvision/csrc/cpu/nms_cpu.cpp  #
  #############################################################################
  # Replace "pass" statement with your code
  device = boxes.device

    # Get the coordinates of the boxes
    x1 = boxes[:, 0]  # Top-left x-coordinate
    y1 = boxes[:, 1]  # Top-left y-coordinate
    x2 = boxes[:, 2]  # Bottom-right x-coordinate
    y2 = boxes[:, 3]  # Bottom-right y-coordinate

    # Compute the area of the boxes
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)

    # Sort the scores in descending order and get their indices
    _, order = scores.sort(0, descending=True)

    keep = []  # List to store the indices of kept boxes

    while order.numel() > 0:
        # Get the index of the current highest score
        i = order[0].item()
        keep.append(i)

        # Compute the intersection area between the current box and the remaining boxes
        xx1 = torch.max(x1[i], x1[order[1:]])
        yy1 = torch.max(y1[i], y1[order[1:]])
        xx2 = torch.min(x2[i], x2[order[1:]])
        yy2 = torch.min(y2[i], y2[order[1:]])

        # Compute the width and height of the intersection area
        w = torch.clamp(xx2 - xx1 + 1, min=0)
        h = torch.clamp(yy2 - yy1 + 1, min=0)

        # Compute the Intersection over Union (IoU)
        intersection = w * h
        iou = intersection / (areas[i] + areas[order[1:]] - intersection)

        # Keep boxes with IoU <= iou_threshold
        remaining = torch.where(iou <= iou_threshold)[0]

        # Update the order list with remaining boxes
        order = order[remaining + 1]  # +1 because we skipped the current highest box

        if topk is not None and len(keep) >= topk:
            break
  #############################################################################
  #                              END OF YOUR CODE                             #
  #############################################################################
  return keep

def ConfScoreRegression(conf_scores, GT_conf_scores):
  """
  Use sum-squared error as in YOLO

  Inputs:
  - conf_scores: Predicted confidence scores
  - GT_conf_scores: GT confidence scores
  
  Outputs:
  - conf_score_loss
  """
  # the target conf_scores for negative samples are zeros
  GT_conf_scores = torch.cat((torch.ones_like(GT_conf_scores), \
                              torch.zeros_like(GT_conf_scores)), dim=0).view(-1, 1)
  conf_score_loss = torch.sum((conf_scores - GT_conf_scores)**2) * 1. / GT_conf_scores.shape[0]
  return conf_score_loss


def BboxRegression(offsets, GT_offsets):
  """"
  Use sum-squared error as in YOLO
  For both xy and wh

  Inputs:
  - offsets: Predicted box offsets
  - GT_offsets: GT box offsets
  
  Outputs:
  - bbox_reg_loss
  """
  bbox_reg_loss = torch.sum((offsets - GT_offsets)**2) * 1. / GT_offsets.shape[0]
  return bbox_reg_loss

