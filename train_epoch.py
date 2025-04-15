import torch

def train_epoch(A_wave, model, loss_criterion, optimizer, train_input, train_target, batch_size, device):
    """
    Trains one epoch with the given data.
    :param training_input: Training inputs of shape (num_samples, num_nodes,
    num_timesteps_train, num_features).
    :param training_target: Training targets of shape (num_samples, num_nodes,
    num_timesteps_predict).
    :param batch_size: Batch size to use during training.
    :return: Average loss for this epoch.
    """
    permutation = torch.randperm(train_input.shape[0])

    epoch_training_losses = []
    for i in range(0, train_input.shape[0], batch_size):
        model.train() # Set model to training mode
        optimizer.zero_grad() # Reset gradients

        indices = permutation[i:i + batch_size]
        X_batch, y_batch = train_input[indices], train_target[indices]
        X_batch = X_batch.to(device=device)
        y_batch = y_batch.to(device=device)

        prediction = model(A_wave, X_batch) # Forward propagation
        loss = loss_criterion(prediction, y_batch)
        loss.backward() # Backpropagation, Compute gradients
        optimizer.step() # Update weights and biases
        epoch_training_losses.append(loss.detach().cpu().numpy())
    return sum(epoch_training_losses)/len(epoch_training_losses)