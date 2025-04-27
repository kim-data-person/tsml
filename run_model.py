import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np

from weighted_adj_mat import get_weighted_adjacency_matrix
from preprocessing import get_spatial_conv_multiplier, get_normalized_node_features, split_dataset
from model import STGCN


def prep_data(input_features, output_feature_index, num_timesteps_input, num_timesteps_output):
    # generate the adjacency matrix and its prefecture names following the matrix row/column order
    W, prefecture_names = get_weighted_adjacency_matrix(sigma=100, epsilon=10e-4)

    # generate the multiplier in the spatial convoltional block
    #spatial_conv_multiplier = get_spatial_conv_multiplier(W)

    # get the model input
    X, means, stds, time_index = get_normalized_node_features(input_features, prefecture_names)

    # split the data into train, val, test
    data = split_dataset(X, time_index, num_timesteps_input, num_timesteps_output, output_feature_index)

    return data, W, means, stds, prefecture_names

def train_model(data, W, num_timesteps_input, num_timesteps_output, dropout_rate, learning_rate, epochs, batch_size, device, verbose=True):

    train_input = data['train_input'] 
    train_target = data['train_target'] 
    val_input = data['val_input'] 
    val_target = data['val_target']


    W = torch.from_numpy(W)
    W = W.to(device=device)

    model = STGCN(W.shape[0], # nodes,
                train_input.shape[3], # features
                num_timesteps_input, 
                num_timesteps_output,
                dropout_rate=dropout_rate).to(device=device) 

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_criterion = nn.MSELoss(reduction='mean')

    training_losses = []
    validation_losses = []

    for epoch in range(epochs):

        permutation = torch.randperm(train_input.shape[0])

        epoch_training_losses = []
        for i in range(0, train_input.shape[0], batch_size):
            model.train() # Set model to training mode
            optimizer.zero_grad() # Reset gradients

            indices = permutation[i:i + batch_size]
            X_batch, y_batch = train_input[indices], train_target[indices]
            X_batch = X_batch.to(device=device)
            y_batch = y_batch.to(device=device)


            prediction = model(W, X_batch) # Forward propagation
            loss = loss_criterion(prediction, y_batch)
            loss.backward() # Backpropagation, Compute gradients
            optimizer.step() # Update weights and biases
            epoch_training_losses.append(loss.detach().cpu().numpy())

        loss = sum(epoch_training_losses)/len(epoch_training_losses)

        training_losses.append(loss)

        # Run validation
        with torch.no_grad(): # Disable gradient calculation for evaluation
            model.eval() # Set model to evaluation mode
            val_input = val_input.to(device=device)
            val_target = val_target.to(device=device)

            prediction = model(W, val_input)
            val_loss = loss_criterion(prediction, val_target).to(device="cpu")
            validation_losses.append(val_loss.detach().numpy().item())

        if epoch % 10 == 0 and verbose:
            print("epochs: ", epoch, 
                "Training loss: {}".format(training_losses[-1]),
                "Validation loss: {}".format(validation_losses[-1]))
            
    return model, training_losses, validation_losses

def evaluate_model(model, data, W, means, stds, output_feature_index, num_timesteps_input, device, prefecture_names):
        
    test_input = data['test_input'] 
    test_target = data['test_target'] 
    test_timestamps = data['test_timestamps']

    W = torch.from_numpy(W)
    W = W.to(device=device)

    # Evaluate the model on the test set
    with torch.no_grad(): # Disable gradient calculation for evaluation
        model.eval() # Set model to evaluation mode
        test_input = test_input.to(device=device)
        test_target = test_target.to(device=device)

        prediction = model(W, test_input)

        prediction_unnormalized = prediction.detach().cpu().numpy()*stds[output_feature_index]+means[output_feature_index]
        target_unnormalized = test_target.detach().cpu().numpy()*stds[output_feature_index]+means[output_feature_index]

    # Convert the segmented timesteps into the original timestamp format

    results = []

    for prefecture_index in range(prediction_unnormalized.shape[1]):

        predictions = []
        targets = []
        timestamps = [] # should be same through all the prefecture_index though

        for i in range(prediction_unnormalized.shape[0]):
            timestamps.append(test_timestamps[num_timesteps_input+i])

            if i==0:
                predictions.append(prediction_unnormalized[i][prefecture_index][0])
                targets.append(target_unnormalized[i][prefecture_index][0])
            elif i==1:
                predictions.append(np.mean([
                    prediction_unnormalized[i][prefecture_index][0],
                    prediction_unnormalized[i-1][prefecture_index][1],
                    ]))
                targets.append(np.mean([
                    target_unnormalized[i][prefecture_index][0],
                    target_unnormalized[i-1][prefecture_index][1],
                    ]))
            else:
                predictions.append(np.mean([
                    prediction_unnormalized[i][prefecture_index][0],
                    prediction_unnormalized[i-1][prefecture_index][1],
                    prediction_unnormalized[i-2][prefecture_index][2],
                    ]))          
                targets.append(np.mean([
                    target_unnormalized[i][prefecture_index][0],
                    target_unnormalized[i-1][prefecture_index][1],
                    target_unnormalized[i-2][prefecture_index][2],
                    ]))
                
        results.append({
            'prefecture_index': prefecture_index,
            'prefecture_name': prefecture_names[prefecture_index],
            'predictions': predictions,
            'targets': targets,
            'timestamps': timestamps
        })

    return results

def run_model(input_features, output_feature_index, num_timesteps_input, num_timesteps_output, dropout_rate, learning_rate, epochs, batch_size, device, verbose=True):

    data, W, means, stds, prefecture_names = prep_data(input_features, output_feature_index, num_timesteps_input, num_timesteps_output)
    model, training_losses, validation_losses = train_model(data, W, num_timesteps_input, num_timesteps_output, dropout_rate, learning_rate, epochs, batch_size, device, verbose)
    results = evaluate_model(model, data, W, means, stds, output_feature_index, num_timesteps_input, device, prefecture_names)
    return results, training_losses, validation_losses

