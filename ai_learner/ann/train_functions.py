import torch
from apex import amp


def to_device(z, device):
    if type(z) == list and len(z) > 1:
        z = [item.to(device) for item in z]
    elif type(z) == list and len(z) == 1:
        z = z[0].to(device)
    else:
        raise ValueError('...')
    return z
#
# def to_device(z, device):
#     return [item.to(device) for item in z]


def unwrap_batch(batch, device):
    x, y = batch
    x = to_device(x, device)
    y = to_device(y, device)
    return x, y


def train_ann(learner, model, loss, optimizer, training_phase, validation_phase,
              callbacks, epochs=100, device='cuda'):

    learner.finish_training = False
    learner.continue_to_next_batch = False
    model.to(device)

    callbacks.training_started()

    for epoch in range(1, epochs + 1):
        callbacks.epoch_started()

        for phase in (training_phase, validation_phase):
            callbacks.phase_started()
            is_training = phase.name == 'training'
            model.train(is_training)

            for batch in phase.loader:
                callbacks.batch_started()

                x, y = unwrap_batch(batch, device)

                with torch.set_grad_enabled(is_training):
                    callbacks.before_forward_pass(gt=y)
                    if learner.continue_to_next_batch:
                        learner.continue_to_next_batch = False
                        print('skipping batch....')
                        continue
                    out = model(x)
                    callbacks.after_forward_pass()

                    loss_score = loss(out, y)

                if is_training:
                    optimizer.zero_grad()
                    if learner.apex:
                        with amp.scale_loss(loss_score, optimizer) as scaled_loss:
                            scaled_loss.backward()
                    else:
                        loss_score.backward()
                    optimizer.step()

                callbacks.batch_ended(phase=phase, pr=out, gt=y, x=x)
            callbacks.phase_ended()
        callbacks.epoch_ended(epoch=epoch)
        if learner.finish_training:
            break
    callbacks.training_ended()
