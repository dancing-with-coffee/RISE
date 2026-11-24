# Adapted from Time-Series-Library (https://github.com/thuml/Time-Series-Library), MIT License.
def adjust_learning_rate(optimizer, scheduler):
    """Advance the learning-rate schedule by one epoch (no-op for a constant rate)."""
    if scheduler is None:
        return
    scheduler.step()
    lr = optimizer.param_groups[0]['lr']
    print('Updating learning rate to {}'.format(lr))
