# --- rdkit_utils.py ---
from rdkit import Chem
from rdkit.Chem import AllChem
from typing import Callable, Optional
from config_loader import Config, RDKitStep
import logging
import functools

logger = logging.getLogger(__name__)

class Hydrogenator:
    def __call__(self, mol: Chem.Mol) -> Chem.Mol:
        return Chem.AddHs(mol, addCoords=True)

class Sanitizer:
    def __call__(self, mol: Chem.Mol) -> Chem.Mol:
        Chem.SanitizeMol(mol)
        return mol

class Kekulizer:
    def __call__(self, mol: Chem.Mol) -> Chem.Mol:
        Chem.Kekulize(mol)
        return mol

class Embedder:
    def __call__(self, mol: Chem.Mol) -> Chem.Mol:
        AllChem.EmbedMolecule(mol, AllChem.ETKDGv2())
        return mol

class Optimizer:
    def __call__(self, mol: Chem.Mol) -> Chem.Mol:
        AllChem.MMFFOptimizeMolecule(mol)
        return mol

RDKIT_STEPS: dict[RDKitStep, Callable[[Chem.Mol], Chem.Mol]] = {
    RDKitStep.HYDROGENATE: Hydrogenator(),
    RDKitStep.SANITIZE: Sanitizer(),
    RDKitStep.KEKULIZE: Kekulizer(),
    RDKitStep.EMBED: Embedder(),
    RDKitStep.OPTIMIZE: Optimizer(),
}

def compose(*functions):
    def composed(mol):
        for func in functions:
            mol = func(mol)
        return mol
    return composed

class RDKitProcessingError(Exception):
    pass

class RDKitKekulizeError(RDKitProcessingError):
    pass

class RDKitMoleculeProcessor:
    def __init__(self, config: Config):
        self.config = config.rdkit_processing
        self.pipeline = self._create_pipeline()

    def _create_pipeline(self) -> Callable[[Chem.Mol], Optional[Chem.Mol]]:
        steps = [RDKIT_STEPS[step] for step in self.config.steps]
        return compose(*steps)

    def process(self, mol: Optional[Chem.Mol]) -> Optional[Chem.Mol]:
        if mol is None:
            return None
        try:
            return self.pipeline(mol)
        except Chem.rdchem.KekulizeException as e:
            raise RDKitKekulizeError(f"Kekulization error: {e}")
        except Exception as e:
            raise RDKitProcessingError(f"Error processing molecule: {e}")

def create_configurable_rdkit_processor(config: Config) -> RDKitMoleculeProcessor:
    return RDKitMoleculeProcessor(config)

def process_molecule(mol_path:str, config: Config) -> Optional[Chem.Mol]:
    processor = create_configurable_rdkit_processor(config)
    mol = Chem.MolFromMolFile(mol_path)
    return processor.process(mol)
