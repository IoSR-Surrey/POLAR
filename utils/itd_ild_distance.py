import torch

def itd_ild_template_distance(estimated, template):

    weigthing_function = template['weigthing_function'][None, None, :]
    max_template_itd = template['itd_max']
    max_template_ild = template['ild_max']
    template_normed_itd = template['itd_normed']
    template_normed_ild = template['ild_normed']

    est_normed_itd = estimated['itd'][:, None, :] / max_template_itd[None, None, :]
    est_normed_ild = estimated['ild'][:, None, :] / max_template_ild[None, None, :]
    
    itd_distance = torch.pow(torch.abs(est_normed_itd - template_normed_itd).clamp(min=1e-8), 0.7)
    ild_distance = torch.pow(torch.abs(est_normed_ild - template_normed_ild).clamp(min=1e-8), 0.7)

    distance_combined = torch.pow(itd_distance * (1 - weigthing_function) + ild_distance * weigthing_function, 1)

    distance = {
        "itd_distance": itd_distance,
        "ild_distance": ild_distance,
        "distance_combined": distance_combined,
    }

    return distance




