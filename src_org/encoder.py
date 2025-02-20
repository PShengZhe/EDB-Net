import copy
import math
import torch
import argparse
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

from transformers import BertTokenizer, BertConfig, BertModel


def euclidean_distance(tensor_x, tensor_y):
    """
    Compute the Euclidean distance between two tensors.

    Args:
        tensor_x (torch.Tensor): A tensor of shape [N, D].
        tensor_y (torch.Tensor): A tensor of shape [M, D].

    Returns:
        torch.Tensor: A Euclidean distance matrix of shape [N, M].
    """
    num_rows_x = tensor_x.size(0)
    num_rows_y = tensor_y.size(0)
    num_features = tensor_x.size(1)
    if num_features != tensor_y.size(1):
        raise Exception

    tensor_x = tensor_x.unsqueeze(1).expand(num_rows_x, num_rows_y, num_features)
    tensor_y = tensor_y.unsqueeze(0).expand(num_rows_x, num_rows_y, num_features)

    return torch.pow(tensor_x - tensor_y, 2).sum(2)


class BertEncoder(nn.Module):
    def __init__(self, args):
        """
        Initialize the BertEncoder module.

        Args:
            args: An object containing various arguments such as file paths and parameters for model configuration.
        """
        super(BertEncoder, self).__init__()

        self.tokenizer = BertTokenizer.from_pretrained(args.fileVocab, do_lower_case=True)
        config = BertConfig.from_json_file(args.fileModelConfig)
        self.bert = BertModel.from_pretrained(args.fileModel, config=config)
        self.attention_layer = copy.deepcopy(self.bert.encoder.layer[11])
        self.linear_layer = nn.Linear(768, 768)
        self.dropout_layer = nn.Dropout(0.1)
        self.la_parameter = args.la
        if args.numFreeze > 0:
            self.freeze_layers(args.numFreeze)
        self.dropout_layer = nn.Dropout(0.1)
        self.alpha_parameter = nn.Parameter(torch.ones(0))  # Weighting coefficient for labels and sentences

    def freeze_layers(self, num_freeze):
        """
        Freeze some layers of the BERT model and unfreeze others.

        Args:
            num_freeze (int): The number of initial layers to freeze.
        """
        unfreeze_layers = ["pooler"]
        for i in range(num_freeze, 12):
            unfreeze_layers.append(f"layer.{i}")

        for name, param in self.bert.named_parameters():
            param.requires_grad = False
            for layer_name in unfreeze_layers:
                if layer_name in name:
                    param.requires_grad = True
                    break

    def forward(self, input_texts, task_classes):
        """
        Forward pass of the BertEncoder.

        Args:
            input_texts (list): A list of input texts.
            task_classes (list): A list of task classes.

        Returns:
            torch.Tensor: The output embeddings.
        """
        combined_task_class = " ".join(task_classes)
        repeated_task_class = [combined_task_class] * len(input_texts)
        sentences = input_texts
        tokenized_input = self.tokenizer(
            sentences,
            text_pair=repeated_task_class,
            truncation="only_first",
            padding=True,
            max_length=250,
            return_tensors='pt'  # Return as PyTorch tensor
        )
        input_ids = tokenized_input['input_ids'].cuda()
        token_type_ids = tokenized_input['token_type_ids'].cuda()
        attention_mask = tokenized_input['attention_mask'].cuda()

        model_outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
        output_embeddings = model_outputs.last_hidden_state[:, 0, :]

        return output_embeddings


class Sampler(nn.Module):
    def __init__(self, args):
        """
        Initialize the Sampler module.

        Args:
            args: An object containing various arguments such as the number of ways, shots, etc.
        """
        super(Sampler, self).__init__()
        self.num_ways = args.numNWay
        self.num_k_shots = args.numKShot
        self.num_q_shots = args.numQShot
        self.feature_dim = 768
        # TOP R
        self.top_k = args.k
        # the number of samples per shot
        self.num_samples_per_shot = args.sample

    def calculate_mean_and_covariance(self, input_features):
        """
        Calculate the mean and covariance of the given features.

        Args:
            input_features (torch.Tensor): Input features.

        Returns:
            tuple: A tuple containing the mean and covariance tensors.
        """
        feature_mean = input_features.mean(dim=1)
        feature_covariances = []
        for i in range(input_features.shape[0]):
            feature_variance = torch.var(input_features[i], dim=0)
            feature_covariances.append(feature_variance)
        feature_covariance_tensor = torch.stack(feature_covariances)

        return feature_mean, feature_covariance_tensor

    def calculate_entropy(self, probability_distribution):
        """
        Calculate the entropy of each query set. Assume that probability_distribution is a probability distribution matrix of shape [batch_size, num_classes].

        Args:
            probability_distribution (torch.Tensor): A probability distribution matrix.

        Returns:
            torch.Tensor: The entropy values of each query set.
        """
        epsilon = 1e-3  # Prevent NaN caused by log(0)
        entropy_value = -torch.sum(probability_distribution * torch.log(probability_distribution + epsilon), dim=-1)
        entropy_value = torch.clamp(entropy_value, min=0.0)
        return entropy_value

    def forward(self, support_embeddings, query_embeddings, classification_results):
        """
        Forward pass of the Sampler.

        Args:
            support_embeddings (torch.Tensor): Support set embeddings.
            query_embeddings (torch.Tensor): Query set embeddings.
            classification_results (torch.Tensor): Classification results.

        Returns:
            tuple: A tuple containing the sampled data and the top-k accuracy.
        """
        similarity_matrix = euclidean_distance(support_embeddings, query_embeddings)
        classification_probabilities = torch.softmax(classification_results, dim=1)
        expanded_probability_tensor = torch.cat([classification_probabilities[:, i:i + 1].repeat(1, self.num_k_shots) for i in range(classification_probabilities.shape[1])], dim=1)
        similarity_probabilities = F.softmax(-similarity_matrix.T, dim=-1)
        similarity_entropy = self.calculate_entropy(similarity_probabilities)
        expanded_entropy = self.calculate_entropy(expanded_probability_tensor)
        combined_similarity = (expanded_probability_tensor.T / (1 + expanded_entropy) + similarity_probabilities.T / (1 + similarity_entropy))
        combined_similarity = combined_similarity.reshape(self.num_ways, 1, -1)

        combined_similarity = combined_similarity.reshape(self.num_ways, self.num_k_shots, -1)
        top_k_values, top_k_indices = combined_similarity.topk(self.top_k, dim=2, largest=True, sorted=True)
        # calculate top R accuracy
        accuracy_scores = []
        for i in range(self.num_ways):
            min_index = i * self.num_q_shots
            max_index = (i + 1) * self.num_q_shots - 1
            for j in range(self.num_k_shots):
                correct_count = 0.0
                for z in range(self.top_k):
                    if top_k_indices[i][j][z] >= min_index and top_k_indices[i][j][z] <= max_index:
                        correct_count += 1
                accuracy_scores.append(correct_count / (self.top_k + 0.0))

        accuracy_tensor = torch.tensor(accuracy_scores)
        mean_accuracy = accuracy_tensor.mean()
        flattened_top_k_indices = top_k_indices.view(-1, self.top_k)

        selected_features = []
        for i in range(flattened_top_k_indices.shape[0]):
            selected_features.append(query_embeddings.index_select(0, flattened_top_k_indices[i]))
        selected_features_tensor = torch.stack(selected_features)
        sampled_data = selected_features_tensor.view(self.num_ways, self.num_k_shots * self.top_k, self.feature_dim)

        return sampled_data, mean_accuracy

    def distribution_calibration(self, query_data, base_mean, base_covariance, alpha=0.21):
        """
        Perform distribution calibration on the query data.

        Args:
            query_data (torch.Tensor): Query data.
            base_mean (torch.Tensor): Base mean.
            base_covariance (torch.Tensor): Base covariance.
            alpha (float, optional): Calibration coefficient. Defaults to 0.21.

        Returns:
            tuple: A tuple containing the calibrated mean and covariance.
        """
        calibrated_mean = (query_data + base_mean) / 2
        calibrated_covariance = base_covariance

        return calibrated_mean, calibrated_covariance


class MyModel(nn.Module):
    def __init__(self, args):
        """
        Initialize the MyModel module.

        Args:
            args: An object containing various arguments for model configuration.
        """
        super(MyModel, self).__init__()
        self.args = args
        self.bert_encoder = BertEncoder(args)
        self.sampler_module = Sampler(args)
        self.first_linear_layer = nn.Linear(768, 256)
        self.classification_layer = nn.Linear(256, self.args.numNWay)
        self.relu_activation = nn.ReLU()

    def forward(self, input_texts, labels):
        """
        Forward pass of the MyModel.

        Args:
            input_texts (list): A list of input texts.
            labels (list): A list of labels.

        Returns:
            tuple: A tuple containing prototypes, query embeddings, accuracy, original prototypes, sampled data, and classification results.
        """
        support_set_size = self.args.numNWay * self.args.numKShot
        query_set_size = self.args.numNWay * self.args.numQShot
        text_embeddings = self.bert_encoder(input_texts, labels)
        classification_logits = self.first_linear_layer(text_embeddings)
        classification_logits = self.relu_activation(classification_logits)
        classification_logits = self.classification_layer(classification_logits)

        support_embeddings = text_embeddings[:support_set_size]
        query_embeddings = text_embeddings[support_set_size:]

        # calculate prototypes
        class_prototypes = support_embeddings.view(self.args.numNWay, -1, support_embeddings.shape[1])
        # calculate original prototypes for generating loss
        original_prototypes = class_prototypes.mean(dim=1)

        # N, S, dim
        sampled_data, accuracy = self.sampler_module(support_embeddings, query_embeddings, classification_logits[support_set_size:])

        combined_prototypes = torch.cat((class_prototypes, sampled_data), dim=1)
        final_prototypes = torch.mean(combined_prototypes, dim=1)

        return (final_prototypes, query_embeddings, accuracy, original_prototypes, sampled_data, classification_logits)

    def visual(self, input_texts):
        """
        Perform a forward pass for visualization purposes.

        Args:
            input_texts (list): A list of input texts.

        Returns:
            tuple: A tuple containing support embeddings, query embeddings, sampled data, and prototypes.
        """
        support_set_size = self.args.numNWay * self.args.numKShot
        query_set_size = self.args.numNWay * self.args.numQShot
        text_embeddings = self.bert_encoder(input_texts)

        support_embeddings = text_embeddings[:support_set_size]
        query_embeddings = text_embeddings[support_set_size:]

        # N, S, dim
        sampled_data, accuracy = self.sampler_module(support_embeddings, query_embeddings)
        sampled_data = sampled_data.cuda()

        class_prototypes = support_embeddings.view(self.args.numNWay, -1, support_embeddings.shape[1])

        combined_prototypes = torch.cat((class_prototypes, sampled_data), dim=1)
        final_prototypes = torch.mean(combined_prototypes, dim=1)

        return support_embeddings, query_embeddings, sampled_data, final_prototypes