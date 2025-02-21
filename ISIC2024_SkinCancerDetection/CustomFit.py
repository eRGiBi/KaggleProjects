

def fit():

    # Focal loss
    
    
    device = "cuda"
    model.to(device)

    # gradient_accumulation_steps = 2

    # for index, batch in enumerate(training_dataloader):
    #     inputs, targets = batch
    #     inputs = inputs.to(device)
    #     targets = targets.to(device)
    #     outputs = model(inputs)
    #     loss = loss_function(outputs, targets)
    #     loss = loss / gradient_accumulation_steps
    #     loss.backward()
    #     if (index + 1) % gradient_accumulation_steps == 0:
    #         optimizer.step()
    #         scheduler.step()
    #         optimizer.zero_grad()