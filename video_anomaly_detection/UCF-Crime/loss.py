import torch
import torch.nn.functional as F

# Define a custom loss function for Multiple Instance Learning (MIL)
def MIL(y_pred, batch_size, is_transformer=0):
    # Initialize loss terms
    loss = torch.tensor(0.).cuda()
    loss_intra = torch.tensor(0.).cuda()
    sparsity = torch.tensor(0.).cuda()
    smooth = torch.tensor(0.).cuda()

    # If not using transformer, reshape the predictions
    if is_transformer==0:
        y_pred = y_pred.view(batch_size, -1)
    else:
        y_pred = torch.sigmoid(y_pred)

    for i in range(batch_size):
         # Randomly permute indices for anomaly and normal instances
        anomaly_index = torch.randperm(30).cuda()
        normal_index = torch.randperm(30).cuda()
        # Extract predictions for anomaly and normal instances
        y_anomaly = y_pred[i, :32][anomaly_index]
        y_normal  = y_pred[i, 32:][normal_index]
        # Calculate maximum and minimum values for anomaly and normal instances
        y_anomaly_max = torch.max(y_anomaly) # anomaly
        y_anomaly_min = torch.min(y_anomaly)
        y_normal_max = torch.max(y_normal) # normal
        y_normal_min = torch.min(y_normal)
        # Calculate the loss component using hinge loss
        loss += F.relu(1.-y_anomaly_max+y_normal_max)
        # Calculate sparsity and smoothness regularization terms
        sparsity += torch.sum(y_anomaly)*0.00008
        smooth += torch.sum((y_pred[i,:31] - y_pred[i,1:32])**2)*0.00008
    # Combine all loss components and divide by batch size
    loss = (loss+sparsity+smooth)/batch_size
    return loss
