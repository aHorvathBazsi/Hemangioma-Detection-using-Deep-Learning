def evaluate_model(model,root,transform_test):
    
    """
    Evaluate the performance of the model on test set.
    
    Please note that in this example we use a test set of 1100 image.
    
    The function will be updated soon.
    """
    evaluation_metrics = {}
    
    test_data = datasets.ImageFolder(os.path.join(root, 'test'), transform=test_transform)
    
    # Evaluate the saved model against the test set
    test_load_all = DataLoader(test_data, batch_size=1100, shuffle=False)
    
    with torch.no_grad():
        correct = 0
        for X_test, y_test in test_load_all:
            y_val = model(X_test)
            predicted = torch.max(y_val,1)[1]
            correct += (predicted == y_test).sum()
        
    print(f'Test accuracy: {correct.item()}/{len(test_data)} = {correct.item()*100/(len(test_data)):7.3f}%')
    ACC = correct.item()*100/(len(test_data))

    arr = confusion_matrix(y_test.view(-1), predicted.view(-1))
    df_cm = pd.DataFrame(arr, class_names, class_names)
    plt.figure(figsize = (9,6))
    sn.heatmap(df_cm, annot=True, fmt="d", cmap='YlGnBu')
    plt.xlabel("prediction")
    plt.ylabel("label (ground truth)")
    plt.show();
    
    True_positive = arr[0,0]
    True_negative = arr[1,1]
    False_positive = arr[1,0]
    False_negative = arr[0,1]

    Sensitivity = True_positive/(True_positive+False_negative)*100
    Specificity = True_negative/(True_negative+False_positive)*100
    PPV = True_positive/(True_positive + False_positive)*100
    NPV = True_negative/(True_negative + False_negative)*100

    evaluation_metrics["TP"] = True_positive
    evaluation_metrics["TN"] = True_negative
    evaluation_metrics["FP"] = False_positive
    evaluation_metrics["FN"] = False_negative
    evalutation_metrics["Sensitivity"] = Sensitivity
    evaluation_metrics["Specificity"] = Specificity
    evaluation_metrics["PPV"] = PPV
    evaluation_metrics["NPV"] = NPV
