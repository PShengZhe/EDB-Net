import torch
from torch.nn import CrossEntropyLoss, BCEWithLogitsLoss
import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer
import torch.nn.functional as F
from sklearn.metrics import precision_score, f1_score, recall_score, accuracy_score, roc_auc_score


def calculate_entropy(probability_distribution):
    """
    Calculate the entropy of each query set. Assume that probability_distribution is a probability distribution matrix of shape [batch_size, num_classes].

    Args:
        probability_distribution (torch.Tensor): A probability distribution matrix.

    Returns:
        torch.Tensor: The entropy values of each query set.
    """
    epsilon = 1e-3  # Prevent NaN caused by log(0)
    entropy = -torch.sum(probability_distribution * torch.log(probability_distribution + epsilon), dim=-1)
    entropy = torch.clamp(entropy, min=0.0)
    return entropy

def euclidean_distance(tensor_x, tensor_y):
    """
    Compute the Euclidean distance between two tensors.

    Args:
        tensor_x (torch.Tensor): A tensor of shape [N, D].
        tensor_y (torch.Tensor): A tensor of shape [M, D].

    Returns:
        torch.Tensor: A Euclidean distance matrix of shape [N, M].
    """
    # tensor_x: N x D
    # tensor_y: M x D
    num_rows_x = tensor_x.size(0)
    num_rows_y = tensor_y.size(0)
    num_features = tensor_x.size(1)
    if num_features != tensor_y.size(1):
        raise Exception
    # Check if tensor_x contains NaN or inf
    if torch.isnan(tensor_x).any() or torch.isinf(tensor_x).any():
        print("tensor_x contains NaN or inf values.")
        tensor_x = torch.nan_to_num(tensor_x)
        tensor_x = torch.clamp(tensor_x, min=-1e10, max=1e10)
    # Check if tensor_y contains NaN or inf
    if torch.isnan(tensor_y).any() or torch.isinf(tensor_y).any():
        print("tensor_y contains NaN or inf values.")
        tensor_y = torch.nan_to_num(tensor_y)
        tensor_y = torch.clamp(tensor_y, min=-1e10, max=1e10)
    tensor_x = tensor_x.unsqueeze(1).expand(num_rows_x, num_rows_y, num_features)
    tensor_y = tensor_y.unsqueeze(0).expand(num_rows_x, num_rows_y, num_features)
    # Limit the range of the result of tensor_x - tensor_y
    difference = torch.clamp(tensor_x - tensor_y, min=-1e10, max=1e10)
    distance_matrix = torch.pow(difference, 2).sum(2)
    return distance_matrix


class LossFunction(torch.nn.Module):
    def __init__(self, args):
        """
        Initialize the loss function class.

        Args:
            args: An object containing training parameters.
        """
        super(LossFunction, self).__init__()
        self.args = args

    def forward(self, model_outputs, label_list):
        """
        Forward propagation function to calculate the loss and evaluation metrics.

        Args:
            model_outputs (tuple): The outputs of the model, including prototypes, query set representations, etc.
            label_list (list or np.ndarray): Labels.

        Returns:
            tuple: A tuple containing the overall loss, precision, recall, F1 score, accuracy, AUC score, and topk accuracy.
        """
        query_set_size = self.args.numNWay * self.args.numQShot
        support_set_size = self.args.numNWay * self.args.numKShot

        class_prototypes, query_representations, topk_accuracy, original_prototypes, sampled_data, ISLB = model_outputs
        label_tensor = torch.tensor(label_list, dtype=float).cuda()
        ISLB_log = F.log_softmax(ISLB, dim=1)
        classification_loss = - ISLB_log * label_tensor
        ISLB_loss = classification_loss.mean()

        distances = euclidean_distance(query_representations, class_prototypes)
        log_probability = F.log_softmax(-distances, dim=1)  # num_query x num_class
        query_labels = label_tensor[support_set_size:]
        query_loss = - query_labels * log_probability
        query_loss = query_loss.mean()

        # calculate generate loss:
        generated_label_indices = torch.tensor(range(self.args.numNWay)).unsqueeze(dim=1)
        generated_label_indices = generated_label_indices.repeat(1, sampled_data.shape[1]).view(-1)
        generated_labels = F.one_hot(generated_label_indices, self.args.numNWay).float().cuda()
        sampled_data = sampled_data.view(-1, sampled_data.shape[2])
        generated_distances = euclidean_distance(sampled_data, original_prototypes)
        generated_log_probability = F.log_softmax(-generated_distances, dim=1)
        sample_loss = - generated_labels * generated_log_probability
        sample_loss = sample_loss.mean()

        # single:
        ISLB_probability = F.softmax(ISLB, dim=1)
        query_classification_probability = ISLB_probability[support_set_size:]
        probability = F.softmax(-distances, dim=1)
        entropy_prob = calculate_entropy(probability)
        entropy_query_class_prob = calculate_entropy(query_classification_probability)
        combined_log_probability = (log_probability.T / (1 + entropy_prob) + query_classification_probability.T / (1 + entropy_query_class_prob)).T
        combined_log_probability = combined_log_probability / 2

        predicted_label_indices = torch.argmax(combined_log_probability, dim=1)

        num_classes = combined_log_probability.shape[1]
        predicted_labels = F.one_hot(predicted_label_indices, num_classes=num_classes).float()
        predicted_prototypes = predicted_labels @ class_prototypes
        false_prototypes = (combined_log_probability * (1 - predicted_labels)) @ class_prototypes
        predicted_false_distances = euclidean_distance(predicted_prototypes, false_prototypes)
        predicted_false_soft = torch.tanh(predicted_false_distances)
        predicted_false_soft = torch.nan_to_num(predicted_false_soft, nan=0.0, posinf=0.0, neginf=0.0)
        pro_loss = 1 - predicted_false_soft.mean()
        print('sam',sample_loss)
        print('cla',ISLB_loss)
        print('pro',pro_loss)
        print('query',query_loss)
        overall_loss = 0.1 * sample_loss + ISLB_loss + pro_loss + query_loss

        evaluation_mode = 'macro'

        query_labels = query_labels.cpu().detach()
        predicted_labels = predicted_labels.cpu().detach()
        precision = precision_score(query_labels, predicted_labels, average=evaluation_mode,zero_division=1)
        recall = recall_score(query_labels, predicted_labels, average=evaluation_mode)
        f1 = f1_score(query_labels, predicted_labels, average=evaluation_mode)
        accuracy = accuracy_score(query_labels, predicted_labels)

        score = F.softmax(-distances, dim=1)
        score = score.cpu().detach()
        auc = roc_auc_score(query_labels, score)

        return overall_loss, precision, recall, f1, accuracy, auc, topk_accuracy