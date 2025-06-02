import torch
import torch.nn as nn
import random

def pgd_attack_random_target(model, images,original_output,eps=0.1, alpha=1e-3, iters=100):
    ori_images = images.detach().clone()
    labels = original_output.argmax(-1)
    num_classes = model(images).shape[1]
    
    # Generate random target labels different from the original prediction
    random_targets = torch.tensor([
        random.choice([i for i in range(num_classes) if i != label.item()])
        for label in labels
    ]).to(images.device)
    
    # Initialize noise
    images = images + torch.empty_like(images).uniform_(-eps, eps)

    for i in range(iters):
        images.requires_grad = True
        outputs = model(images)

        model.zero_grad()
        cost = nn.CrossEntropyLoss()(outputs, random_targets)
        cost.backward()

        adv_images = images - alpha * images.grad.sign()
        eta = torch.clamp(adv_images - ori_images, min=-eps, max=eps)
        images = (ori_images + eta).detach()

    return images


def pgd_attack_non_target(model, images, eps=0.1, alpha=1e-3, iters=100):
    ori_images = images.detach().clone()
    labels = model(images).argmax(-1)
    
    # Initialize noise
    images = images + torch.empty_like(images).uniform_(-eps, eps)

    for i in range(iters):
        images.requires_grad = True
        outputs = model(images)

        model.zero_grad()
        cost = nn.CrossEntropyLoss()(outputs, labels)
        cost.backward()

        adv_images = images + alpha * images.grad.sign()
        eta = torch.clamp(adv_images - ori_images, min=-eps, max=eps)
        images = (ori_images + eta).detach()

    return images