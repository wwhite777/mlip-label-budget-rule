#!/usr/bin/env python3
"""Offline ORB v3 conservative-inf-mpa energy-only fine-tuning."""
import torch
from finetune_common import parser, run


class OrbBackend:
    name = 'ORB v3 conservative-inf-mpa'

    def __init__(self, a, atoms):
        from orb_models.forcefield import atomic_system, base, pretrained
        self.model = pretrained.orb_v3_conservative_inf_mpa(
            weights_path=str(a.checkpoint.resolve()), device=a.device,
            precision='float32-highest', compile=False, train=True)
        self.device, self.base = a.device, base
        self.graphs = {m: atomic_system.ase_atoms_to_atom_graphs(atom, self.model.system_config,
                                                               device='cpu') for m,atom in atoms.items()}

    def e0_parameter(self):
        return self.model.heads[self.model.energy_name].reference.linear.weight

    def original_e0(self):
        return dict(enumerate(self.e0_parameter().detach().cpu().reshape(-1).tolist()))

    def set_e0(self, values):
        with torch.no_grad():
            self.e0_parameter().copy_(torch.tensor([values[z] for z in range(118)],
                                                   device=self.device).reshape_as(self.e0_parameter()))

    def configure_trainable(self, variant, e0='residual'):
        for name, p in self.model.named_parameters():
            p.requires_grad_(variant == 'full' or name.startswith('heads.' + self.model.energy_name + '.'))
        if e0 == 'foundation':
            self.model.heads[self.model.energy_name].reference.requires_grad_(False)

    def energy(self, mids, training):
        batch = self.base.batch_graphs([self.graphs[m] for m in mids]).to(self.device)
        # ConservativeForcefieldRegressor.forward computes forces/stress unconditionally.
        # Use its exact energy component, preserving the ZBL pair term.
        batch.edge_features['vectors'] = batch.compute_differentiable_edge_vectors()[0]
        node = self.model.model(batch)['node_features']
        energy = self.model.heads[self.model.energy_name].predict(node, batch)
        if self.model.pair_repulsion:
            energy = energy + self.model.pair_repulsion_fn(batch)['energy']
        return energy.reshape(-1)


if __name__ == '__main__':
    run(parser(__doc__).parse_args(), OrbBackend)
