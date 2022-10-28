import numpy as np
import torch

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def critic_attack(batch_obs,critic_model,eps=0.1,search_space=None):
    
    batch_adv_obs = []

    for obs in batch_obs:

        obs = torch.as_tensor(np.expand_dims(obs,axis=0), dtype=torch.float32)

        obs_shape = obs.shape
        obs = obs.flatten()
        adv_obs = torch.autograd.Variable(obs.clone(), requires_grad=True)
        obs_size = len(obs)
        
        if search_space is None:
            search_space = torch.ones((obs_size))
        assert search_space.shape == (obs_size,)
        search_space = search_space.clone()
        
        output = critic_model(adv_obs.reshape(obs_shape).to(device)).flatten()
        
        num_classes = output.size()[0]
        
        jacobian = torch.zeros(num_classes, *adv_o.size())
        grad_output = torch.zeros(*output.size())
        if adv_o.is_cuda:
            grad_output = grad_output.cuda()
            jacobian = jacobian.cuda()
        for i in range(num_classes):
            if adv_o.grad is not None:
                adv_o.grad.zero_()
            grad_output.zero_()
            grad_output[i] = 1
            output.backward(grad_output.to(device), retain_graph=True)
            jacobian[i] = adv_o.grad.data

        saliency_map = torch.mul(jacobian, search_space.float()).detach().numpy()

        saliency_eps = np.sum(np.abs(saliency_map))

        if saliency_eps < 1e-9:
            saliency_map = saliency_map + (eps/saliency_map.size) 
            saliency_eps = np.sum(np.abs(saliency_map))

        perturbation_map = - saliency_map * (eps / saliency_eps)
        
        adv_obs = adv_obs + torch.Tensor(perturbation_map)


        batch_adv_obs.append( adv_obs.flatten() )
        
    batch_adv_obs = torch.stack(batch_adv_obs)
        
    return batch_adv_obs