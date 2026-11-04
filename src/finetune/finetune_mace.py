#!/usr/bin/env python3
"""Offline MACE-MP-0 medium energy-only fine-tuning; see FINETUNE_RECIPE.md."""
import torch
from finetune_common import parser, run


class MaceBackend:
    name = 'MACE-MP-0 medium'

    def __init__(self, a, atoms):
        from mace import data, modules
        from mace.tools import AtomicNumberTable, torch_geometric
        from mace.tools.scripts_utils import extract_config_mace_model
        from mace.tools.finetuning_utils import load_foundations_elements
        foundation = torch.load(a.checkpoint, map_location='cpu', weights_only=False).float()
        cfg = extract_config_mace_model(foundation)
        self.zs = [int(z) for z in foundation.atomic_numbers]
        self.table = AtomicNumberTable(self.zs)
        # This is the foundation reconstruction/transfer path used by configure_model.
        model = modules.ScaleShiftMACE(**cfg)
        model = load_foundations_elements(model, foundation, self.table, load_readout=True,
                                          max_L=cfg['hidden_irreps'].lmax, default_dtype=torch.float32)
        # Preserve ALL checkpoint species and its scale/shift; replace E0 only after transfer,
        # whose final state_dict copy would otherwise overwrite a newly estimated E0.
        original = model.atomic_energies_fn.atomic_energies.detach().clone()
        del model.atomic_energies_fn._buffers['atomic_energies']
        model.atomic_energies_fn.register_parameter('atomic_energies', torch.nn.Parameter(original))
        self.model = model.to(a.device)
        self.device, self.geo = a.device, torch_geometric
        keys = data.KeySpecification(info_keys={}, arrays_keys={})
        self.graphs = {}
        for mid, atom in atoms.items():
            config = data.config_from_atoms(atom, key_specification=keys, head_name=model.heads[0])
            self.graphs[mid] = data.AtomicData.from_config(config, z_table=self.table,
                                                          cutoff=float(model.r_max), heads=model.heads)

    def e0_parameter(self):
        return self.model.atomic_energies_fn.atomic_energies

    def original_e0(self):
        return dict(zip(self.zs, self.e0_parameter().detach().cpu().reshape(-1).tolist()))

    def set_e0(self, values):
        with torch.no_grad():
            self.e0_parameter().copy_(torch.tensor([values[z] for z in self.zs], device=self.device).reshape_as(self.e0_parameter()))

    def configure_trainable(self, variant, e0='residual'):
        for name, p in self.model.named_parameters():
            p.requires_grad_(variant == 'full' or name.startswith('readouts.') or name == 'atomic_energies_fn.atomic_energies')
        if e0 == 'foundation':
            self.model.atomic_energies_fn.requires_grad_(False)

    def energy(self, mids, training):
        batch = self.geo.Batch.from_data_list([self.graphs[m] for m in mids]).to(self.device)
        return self.model(batch.to_dict(), training=training, compute_force=False,
                          compute_virials=False, compute_stress=False)['energy'].reshape(-1)


if __name__ == '__main__':
    run(parser(__doc__).parse_args(), MaceBackend)
