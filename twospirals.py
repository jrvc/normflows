import torch
import torch.nn as nn
from torch.distributions import Normal
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt

class TwoSpirals(Dataset):
    def __init__(self, num_points):
        self.num_points = num_points
        self._create_data()

    def __getitem__(self, item):
        return self.data[item]

    def __len__(self):
        return self.num_points

    def _create_data(self):
        n = torch.sqrt(torch.rand(self.num_points // 2)) * 540 * (2 * np.pi) / 360
        d1x = -torch.cos(n) * n + torch.rand(self.num_points // 2) * 0.5
        d1y = torch.sin(n) * n + torch.rand(self.num_points // 2) * 0.5
        x = torch.cat([torch.stack([d1x, d1y]).t(), torch.stack([-d1x, -d1y]).t()])
        self.data = x / 3 + torch.randn_like(x) * 0.1

train_loader = DataLoader(TwoSpirals(128000), batch_size=128, shuffle=True)
test_loader = DataLoader(TwoSpirals(128000), batch_size=128, shuffle=False)



def plot_samples(s=None):
  d = test_loader.dataset.data.numpy()
  fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(26, 12))
  ax[0].axis('off'); ax[1].axis('off')
  ax[0].set_title('Data', fontsize=24); ax[1].set_title('Samples', fontsize=24)
  ax[0].hist2d(d[...,0], d[...,1], bins=256, range=[[-4, 4], [-4, 4]])
  if s is not None:
    s = s.detach().cpu().numpy()
    ax[1].hist2d(s[...,0], s[...,1], bins=256, range=[[-4, 4], [-4, 4]])
  plt.show()

#plot_samples()

class Flow(nn.Module):
    def __init__(self,bijections):
        super().__init__()
        self.bijections = nn.ModuleList(bijections)

    @property
    def base_dist(self):
        return Normal(
            loc=torch.zeros(2, device=device),
            scale=torch.ones(2, device=device),
        )

    def log_prob(self, x):
        # log(p(x)) = log(base_dist(z)) + det|Jacobian|
        # z = f⁻¹(x)
        for bijection in self.bijections:
            inv_biject = invert(bijection)
            z = inv_biject(x)
            x = z # update x for applyting next bijection
            log_prob += log(base_dist((z,)) 

        return log_prob

    def sample(self, num_samples):
        # z ~ base_dist(z)
        # x = f(z)
        z = base_dist((num_samples,))
        for bijection in self.bijections:

        return z
