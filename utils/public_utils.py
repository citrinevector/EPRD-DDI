def cal_torch_model_params(model):
    """Return the total and trainable parameter counts of a model."""
    total_params = sum(p.numel() for p in model.parameters())
    total_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {'total_params': total_params, 'total_trainable_params': total_trainable_params}
