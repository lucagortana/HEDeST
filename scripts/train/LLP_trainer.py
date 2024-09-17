import torch
import os

def evaluate(model, dataloader, agg_loss = 'mean', alpha = 0.5, device=torch.device("cpu")):
    model.eval()  # Set model to evaluation mode
    running_loss = 0.0
    with torch.no_grad():
        for images_list, true_proportions in dataloader: # images -> images_list
            spot_outputs = [] #created spot_outputs
            true_proportions = true_proportions.to(device)
            
            for images in images_list: #created loop over images_list
                images = images.to(device)
                outputs = model(images)
                spot_outputs.append(outputs)
                
            outputs = torch.cat(spot_outputs, dim=0) #concatenated spot_outputs
            
            loss = model.loss_comb(outputs, true_proportions, agg=agg_loss, alpha=alpha)
            running_loss += loss.item()
    
    avg_loss = running_loss / len(dataloader)
    return avg_loss

def train(model, train_loader, val_loader, test_loader, optimizer, agg_loss='mean', alpha=0.5, num_epochs=25, out_dir='models'):
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on {device}.")
    
    best_val_loss = float('inf')
    best_model_state = None
    
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        
        for images_list, true_proportions in train_loader: # images -> images_list
            optimizer.zero_grad()
            spot_outputs = [] #created spot_outputs
            true_proportions = true_proportions.to(device)
            
            for images in images_list: #created loop over images_list
                images = images.to(device)
                outputs = model(images)
                spot_outputs.append(outputs)
                
            outputs = torch.cat(spot_outputs, dim=0) #concatenated spot_outputs
            
            loss = model.loss_comb(outputs, true_proportions, agg=agg_loss, alpha=alpha)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
        
        train_loss = running_loss / len(train_loader)
        val_loss = evaluate(model, val_loader, agg_loss=agg_loss, alpha=alpha, device=device)
        
        print(f'Epoch {epoch + 1}/{num_epochs}, Train Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}')
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict()
            best_model_path = os.path.join(out_dir, 'best_model.pth')
            torch.save(best_model_state, best_model_path)
            print(f'Validation loss improved. Saving best model at {best_model_path} (epoch {epoch + 1}).')
    
    if val_loss != best_val_loss:
        final_model_state = model.state_dict()
        final_model_path = os.path.join(out_dir, 'final_model.pth')
        torch.save(final_model_state, final_model_path)
        print("Best model and final model are different. Both models have been saved.")
    else:
        print("Best model and final model are the same. Only the final model has been saved.")
    
    print('Training complete. Evaluating on test set...')
    
    test_loss = evaluate(model, test_loader, agg_loss=agg_loss, alpha=alpha, device=device)
    print(f'Test Loss on final model: {test_loss:.4f}')

    best_model = type(model)(*model.args, **model.kwargs)  # Create a new model instance with the same architecture
    best_model.load_state_dict(torch.load(best_model_path))
    best_model = best_model.to(device)

    test_loss_best = evaluate(best_model, test_loader, agg_loss=agg_loss, alpha=alpha, device=device)
    print(f'Test Loss on best model: {test_loss_best:.4f}')