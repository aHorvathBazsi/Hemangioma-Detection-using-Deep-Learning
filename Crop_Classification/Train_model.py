def train_model(model,criterion,optimizer,scheduler,number_epochs,train_loader,test_loader, model_save = True):
    """
    Input:
     - model: the created model (in this example CNN_model class)
     - criterion: Loss function defined by user (in this example CrossEntropyLoss)
     - optimizer: Optimizer used for training (in this example Adam optimizer with initial learning rate of 0.01)
     - scheduler: Learning-rate decay (in this example ExponentialLR with gamma = 0.95)
     - number_epochs: number of epochs of training
     - train/test loader: DataLoaders created with create_datasets function
     - model_save: boolean that controls whether you want to save the state dictionary after each epoch or not
     
    Output:
     - 
     Please note that the function was created for a batch size of 50, train set of 2000 images and test set of 1064 images
     More generalized implementation will be updated soon!
    """
    
    start_time = time.time()
    
    train_output = {}
    # global training metrics
    train_losses = []
    test_losses = []
    train_correct = []
    test_correct = []
    
    for i in range(epochs):
        #metrics for each epoch
        trn_corr = 0
        tst_corr = 0
        trn_loss = 0
        tst_loss = 0
        
        # print learning rate to verify learning rate decay
        for param_group in optimizer.param_groups:
            print('learing rate for epoch {} is {}'.format(i,param_group['lr']))
            
        for b, (X_train, y_train) in enumerate(train_loader):
        
            b+=1
        
            # Apply the model
            y_pred = model(X_train)
            loss = criterion(y_pred, y_train)
            trn_loss += loss
 
            # Tally the number of correct predictions
            predicted = torch.max(y_pred.data, 1)[1]
            batch_corr = (predicted == y_train).sum()
            trn_corr += batch_corr
        
            # Update parameters
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Print interim results
            if b%5 == 0:
                print(f'epoch: {(i+1):2}  batch: {b:4} [{50*b:6}/2000]  loss: {loss.item():10.8f}  \accuracy: {trn_corr.item()*100/(50*b):7.3f}%')
        
        #append loss and number of correct predictions
        train_losses.append(trn_loss/b)
        train_correct.append(trn_corr)

        # Run the testing batches
        with torch.no_grad():
            for b, (X_test, y_test) in enumerate(test_loader):

            # Apply the model
                y_val = CNNmodel(X_test)
                loss = criterion(y_val,y_test)
                tst_loss += loss

            # Tally the number of correct predictions
                predicted = torch.max(y_val.data, 1)[1] 
                tst_corr += (predicted == y_test).sum()
            
        loss = criterion(y_val, y_test)
        test_losses.append(tst_loss/b)
        test_correct.append(tst_corr)
        
        #print validation loss and accuracy for each epoch
        print(f'epoch: {i:2}  val_loss: {loss.item():10.8f}  val_accuracy: {tst_corr.item()*100/(1064):7.3f}%')
        
        #decrease learning rate
        scheduler.step()

        if model_save:
            torch.save(CNNmodel.state_dict(), 'Hemagioma_dataset2_epoch_'+'{}'.format(i+1)+'.pt')
    
    print(f'\nDuration: {time.time() - start_time:.0f} seconds') # print the time elapsed

    train_output["train_losses"] = train_losses
    train_output["test_losses"] = test_losses
    train_output["train_correct"] = train_correct
    train_output["test_correct"] = test_correct
    
    return train_output
