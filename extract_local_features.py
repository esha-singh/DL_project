import torch
import torch.nn.functional as F

def delf_extract(image, model):
    attn_threshold = 0.05
    with torch.no_grad():
        _, _, _, _, encoded_f, _, _, prob = model(image, labels=None, global_only=False, training=False)
    
    locations = []
    scores = []
    descriptors = []
    ii = 0
    #print(prob.shape)
    for i in range(prob.shape[2]):
        for j in range(prob.shape[3]):
            #print(prob[:, :, i, j])
            if prob[0, :, i, j].detach().cpu() > attn_threshold:
                locations.append((i, j))
                scores.append(prob[0, :, i, j].detach().cpu())
                ii += 1

    
    for loc in locations:
        #encoded_f = F.normalize(encoded_f)
        descriptors.append(encoded_f[0, :, loc[0], loc[1]].detach().cpu().numpy())

    return locations, descriptors, scores
                
    