def show_batch_of_images(DataLoader):
    """
    Plots a batch of data from the given DataLoader (train or test)
    """
    
    # Grab the first batch of 30 images
    for images,labels in train_loader: 
        break

    # Print the labels
    print('Label:', labels.numpy())
    print('Class:', *np.array([class_names[i] for i in labels]))

    im = make_grid(images, nrow=5)  # the default nrow is 8

    # Inverse normalize the images
    inv_normalize = transforms.Normalize(
        mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
        std=[1/0.229, 1/0.224, 1/0.225]
    )
    im_inv = inv_normalize(im)

    # Print the images
    plt.figure(figsize=(40,12))
    plt.imshow(np.transpose(im_inv.numpy(), (1, 2, 0)));
