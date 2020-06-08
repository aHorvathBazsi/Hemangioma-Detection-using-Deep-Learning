def create_datasets(root,batch_size_train,batch_size_test,transform_train,transform_test):
    """
    Creating train/test set based on a given root/path and a DataLoader with allows to feed batches of data.
    
    Input : root, barch_size_train, batch_size_test, transform_train,transform_test
    Output : train_loader,test_loader, class_names (the names of classes used in binary classification)
    """
    train_data = datasets.ImageFolder(os.path.join(root, 'train'), transform=train_transform)
    test_data = datasets.ImageFolder(os.path.join(root, 'test'), transform=test_transform)

    train_loader = DataLoader(train_data, batch_size=batch_size_train, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=batch_size_test, shuffle=True)

    class_names = train_data.classes

    print(class_names)
    print(f'Training images available: {len(train_data)}')
    print(f'Testing images available:  {len(test_data)}')
    
    return (train_loader,test_loader,class_names)
