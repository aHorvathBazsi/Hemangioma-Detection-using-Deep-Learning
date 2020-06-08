def count_parameters(model):
    """
    Calculates the number of parameters for a given model.
    """
    params = [p.numel() for p in model.parameters() if p.requires_grad]
    for item in params:
        print(f'{item:>8}')
    print(f'________\n{sum(params):>8}')
