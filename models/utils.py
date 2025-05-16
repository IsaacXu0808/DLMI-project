import torch

def model_info(model, input_size=(1, 768, 14, 14)):
    """
    Prints model parameter count and estimated model size in MB.
    
    Args:
        model (nn.Module): The model instance.
        input_size (tuple): Input size to the model (default is ViT's output).

    Returns:
        param_count (int): Total number of parameters.
        model_size (float): Model size in MB.
    """
    model = model.cpu()
    
    param_count = sum(p.numel() for p in model.parameters())
    
    model_size = param_count * 4 / (1024 ** 2)
    
    print(f"Model Parameters: {param_count:,}")
    print(f"Estimated Model Size: {model_size:.2f} MB")
    
    with torch.no_grad():
        dummy_input = torch.randn(input_size)
        output = model(dummy_input)
        print(f"Output Shape: {output.shape}")
    
    return param_count, model_size