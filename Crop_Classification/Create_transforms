def create_transforms(rotation_angle,flip_probability,crop_size):
    
    transforms_dict = {}
    
    """
    Function used create transforms for train/test datasets. These transforms will be used in the dataset creation step.
    Please note that rotation and flip are applied only for the training set because we don't need augmentation on the test set.
    Resizing, cropping, normalization and ToTensor is applied to both sets because we want to have the same distribution for train/test sets.
    """
    train_transform = transforms.Compose([
        transforms.RandomRotation(rotation_angle),      # rotate +/- rotation_angle degrees
        transforms.RandomHorizontalFlip(),              # reverse flip_probability% of images
        transforms.Resize(crop_size),                   # resize shortest side to crop_size pixels
        transforms.CenterCrop(crop_size),               # crop longest side to crop_size pixels at center
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

    test_transform = transforms.Compose([
        transforms.Resize(crop_size),
        transforms.CenterCrop(crop_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])
    
    transforms_dict["train"] = train_transform
    transforms_dict["test"] = test_transform
    
    return transforms_dict
