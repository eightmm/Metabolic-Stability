import torch, dgl, numpy, random

def train_model( model, loader, optimizer, scheduler, device='cpu' ):
    model.train()
    loss_function = torch.nn.BCEWithLogitsLoss()
    model_loss = 0.0

    for g, label, n in loader:
        pred_value = model( g.to(device) )
        true_value = label.to(device)

        loss = loss_function( pred_value, true_value )

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        model_loss += loss.item() * g.batch_size

    scheduler.step()
    divisor = len(loader.dataset)

    return model_loss / divisor

def valid_model( model, loader, device='cpu' ):
    model.eval()
    loss_function = torch.nn.BCEWithLogitsLoss()
    model_loss = 0.0

    pred = []
    true = []
    name = []
    with torch.no_grad():
        for g, label, n in loader:
            pred_value = model( g.to(device) )
            true_value = label.to(device)

            loss = loss_function( pred_value, true_value )

            model_loss += loss.item() * g.batch_size

            pred_value = torch.sigmoid(pred_value)

            pred.extend( [ x.item() for x in pred_value ] )
            true.extend( [ x.item() for x in true_value ] )
            name.extend( [ x for x in n ] )

    divisor = len(loader.dataset)

    return model_loss / divisor, pred, true, name
