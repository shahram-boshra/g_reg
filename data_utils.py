# --- data_utils.py ---
from rdkit_utils import process_molecule
import torch
import torch.nn.functional as F
from rdkit import Chem
import pandas as pd
from pathlib import Path
import json
import os
import diskcache
from typing import Dict, List, Optional
import torch_geometric.data
from config_loader import Config
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class OneHotEncoder:
    @staticmethod
    def encode(value: int, range_size: int) -> torch.Tensor:
        return F.one_hot(torch.tensor(value), num_classes=range_size).float()

class MoleculeFeatureExtractor:
    @staticmethod
    def get_atomic_features_one_hot(mol: Chem.Mol, feature_ranges: Dict[str, int]) -> torch.Tensor:
        atomic_features = []
        formal_charges = []
        for atom in mol.GetAtoms():
            atomic_num = atom.GetAtomicNum()
            hybridization = int(atom.GetHybridization())
            degree = atom.GetDegree()
            formal_charge = atom.GetFormalCharge()
            formal_charges.append(formal_charge)
            chiral_tag = int(atom.GetChiralTag())
            implicit_valence = atom.GetImplicitValence()
            num_h = atom.GetTotalNumHs()

            atom_features = [
                OneHotEncoder.encode(atomic_num, feature_ranges["atomic_nums"]),
                OneHotEncoder.encode(hybridization, feature_ranges["hybridizations"]),
                OneHotEncoder.encode(degree, feature_ranges["degrees"]),
                OneHotEncoder.encode(MoleculeFeatureExtractor.shift_formal_charge(formal_charge, formal_charges), feature_ranges["formal_charges"]),
                OneHotEncoder.encode(chiral_tag, feature_ranges["chiral_tags"]),
                OneHotEncoder.encode(implicit_valence, feature_ranges["implicit_valences"]),
                OneHotEncoder.encode(num_h, feature_ranges["num_h_list"]),
            ]
            atomic_features.append(torch.cat(atom_features))
        return torch.stack(atomic_features)

    @staticmethod
    def get_bond_features_one_hot(mol: Chem.Mol) -> torch.Tensor:
        bond_features = []
        for bond in mol.GetBonds():
            bond_type = bond.GetBondType()
            is_conjugated = bond.GetIsConjugated()
            is_in_ring = bond.IsInRing()
            bond_stereo = int(bond.GetStereo())

            bond_features.append(torch.tensor([
                1 if bond_type == Chem.BondType.SINGLE else 0,
                1 if bond_type == Chem.BondType.DOUBLE else 0,
                1 if bond_type == Chem.BondType.TRIPLE else 0,
                1 if bond_type == Chem.BondType.AROMATIC else 0,
                1 if is_conjugated else 0,
                1 if is_in_ring else 0,
                bond_stereo,
            ], dtype=torch.float))
        return torch.stack(bond_features) if bond_features else torch.empty(0, 7)

    @staticmethod
    def shift_formal_charge(formal_charge, formal_charges):
        min_formal_charge = min(formal_charges)
        offset = abs(min_formal_charge) if min_formal_charge < 0 else 0
        return formal_charge + offset

class MoleculeProcessor:
    def __init__(self, root: str, directed: bool = False, rdkit_config: Config = None):
        self.root = root
        self.directed = directed
        self.raw_loader = RawDataLoader(root)
        self.data_handler = ProcessedDataHandler(root)
        self.feature_ranges = self.data_handler.load_feature_ranges(self.raw_loader.load_raw_files())
        self.rdkit_config = rdkit_config

    def load_molecule(self, mol_path: str) -> Optional[Chem.Mol]:
        return process_molecule(mol_path, config = self.rdkit_config)

    def process_molecule(self, mol: Chem.Mol) -> Optional[torch_geometric.data.Data]:
        return self.mol_to_graph(mol, self.feature_ranges, self.directed)

    @staticmethod
    def mol_to_graph(mol: Chem.Mol, feature_ranges: Dict[str, int], directed: bool = False) -> Optional[torch_geometric.data.Data]:
        if mol is None:
            return None
        x = MoleculeFeatureExtractor.get_atomic_features_one_hot(mol, feature_ranges)
        edge_attr_onehot = MoleculeFeatureExtractor.get_bond_features_one_hot(mol)
        edge_index_list = []
        for bond in mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()
            edge_index_list.append([i, j])
            if not directed:
                edge_index_list.append([j, i])
        edge_index = torch.tensor(edge_index_list, dtype=torch.long).t().contiguous()
        pos = mol.GetConformer().GetPositions()
        pos = torch.tensor(pos, dtype=torch.float)
        graph = torch_geometric.data.Data(x=x, edge_index=edge_index, edge_attr=edge_attr_onehot, pos=pos)
        return graph

class RawDataLoader:
    def __init__(self, root: str):
        self.root = Path(root)
        self.mol_dir = self.root / 'Mols'

    def load_raw_files(self) -> List[str]:
        files = [str(f) for f in self.mol_dir.glob('*.mol')]
        return files

class FeatureRangeCalculator:
    @staticmethod
    def calculate_feature_ranges(mol_paths: List[str]) -> Dict[str, int]:
        data = []
        formal_charges = []

        for mol_path in mol_paths:
            mol = Chem.MolFromMolFile(mol_path)
            if mol is not None:
                for atom in mol.GetAtoms():
                    data.append({
                        "atomic_num": atom.GetAtomicNum(),
                        "hybridization": int(atom.GetHybridization()),
                        "degree": atom.GetDegree(),
                        "formal_charge": atom.GetFormalCharge(),
                        "chiral_tag": int(atom.GetChiralTag()),
                        "implicit_valence": atom.GetImplicitValence(),
                        "num_h": atom.GetTotalNumHs()
                    })
                    formal_charges.append(atom.GetFormalCharge())

        df = pd.DataFrame(data)

        feature_ranges = {
            "atomic_nums": int(df["atomic_num"].max()) + 1 if not df.empty else 1,
            "hybridizations": int(df["hybridization"].max()) + 1 if not df.empty else 1,
            "degrees": int(df["degree"].max()) + 1 if not df.empty else 1,
            "formal_charges": max(formal_charges) + abs(min(formal_charges)) + 1 if formal_charges else 1,
            "chiral_tags": int(df["chiral_tag"].max()) + 1 if not df.empty else 1,
            "implicit_valences": int(df["implicit_valence"].max()) + 1 if not df.empty else 1,
            "num_h_list": int(df["num_h"].max()) + 1 if not df.empty else 1,
        }
        return feature_ranges

class ProcessedDataHandler:
    def __init__(self, root: str, use_cache: bool = True, cache_expiry: int = 3600):
        self.root = Path(root)
        self.use_cache = use_cache
        self.cache_dir = self.root / 'processed_graphs_cache'
        self.processed_dir = self.root / 'processed'
        self.feature_ranges_path = self.root / 'feature_ranges.json'
        self.cache_expiry = cache_expiry
        if self.use_cache:
            os.makedirs(self.cache_dir, exist_ok=True)
            self.cache = diskcache.Cache(self.cache_dir)
        else:
            self.cache = None
        os.makedirs(self.processed_dir, exist_ok=True)

    def save_feature_ranges(self, feature_ranges: Dict[str, int]) -> None:
        with open(self.feature_ranges_path, 'w') as f:
            json.dump(feature_ranges, f)

    def load_feature_ranges(self, raw_file_names: List[str]) -> Dict[str, int]:
        if os.path.exists(self.feature_ranges_path):
            with open(self.feature_ranges_path, 'r') as f:
                ranges = json.load(f)
                return ranges
        else:
            feature_ranges = FeatureRangeCalculator.calculate_feature_ranges(raw_file_names)
            self.save_feature_ranges(feature_ranges)
            return feature_ranges

    def save_graph(self, graph: torch_geometric.data.Data, file_name: str) -> None:
        processed_path = self.processed_dir / file_name
        if self.use_cache and self.cache:
            self.cache.set(file_name, graph, expire=self.cache_expiry)
        else:
            torch.save(graph, processed_path)

    def load_graph(self, file_name: str) -> Optional[torch_geometric.data.Data]:
        processed_path = self.processed_dir / file_name
        if self.use_cache and self.cache:
            graph = self.cache.get(file_name)
            if graph is not None:
                return graph
            else:
                return None
        else:
            try:
                torch.serialization.add_safe_globals([
                    torch_geometric.data.data.DataEdgeAttr,
                    torch_geometric.data.data.DataTensorAttr,
                    torch_geometric.data.storage.GlobalStorage
                ])
                graph = torch.load(processed_path)
                return graph
            except FileNotFoundError:
                return None

class DatasetError(Exception):
    pass

class EmptyDatasetError(DatasetError):
    pass

class ColumnNotFoundError(DatasetError):
    def __init__(self, column_name: str) -> None:
        super().__init__(f"Column '{column_name}' not found in CSV.")

class DataProcessor:
    def __init__(self, root: str, use_cache: bool = True, rdkit_config: Config = None):
        self.root = root
        self.use_cache = use_cache
        self.raw_loader = RawDataLoader(root)
        self.data_handler = ProcessedDataHandler(root, use_cache)
        self.molecule_processor = MoleculeProcessor(root, rdkit_config=rdkit_config)
        self.rdkit_config = rdkit_config

    def _load_molecule_data(self, raw_path: str) -> Optional[Chem.Mol]:
        return self.molecule_processor.load_molecule(raw_path)

    def _process_molecule_graph(self, mol: Chem.Mol, mol_name: str, processed_name: str, target_df: pd.DataFrame) -> None:
        if mol is not None:
            graph = self.molecule_processor.process_molecule(mol)
            if graph is not None:
                try:
                    target = target_df.loc[(mol_name,)].values
                    target = torch.tensor(target, dtype=torch.float).reshape(1, -1)
                    graph.y = target
                    self.data_handler.save_graph(graph, processed_name)
                except KeyError as e:
                    raise ColumnNotFoundError(f'Molecule name {mol_name} not found or index error in target file: {e}')
                except Exception as e:
                    raise DatasetError(f'Error saving graph {processed_name}: {e}')
            else:
                logger.warning(f'Trouble processing graph {processed_name} or empty graph')
        else:
            logger.warning(f'MOL file {mol_name}.mol not found or corrupted, or rdkit processing failed')

    def process_data(self, target_df: pd.DataFrame) -> None:
        raw_file_names = self.raw_loader.load_raw_files()
        if not raw_file_names:
            raise EmptyDatasetError("No MOL files found in the specified directory.")
        for raw_path, processed_name in zip(raw_file_names, [Path(f).name.replace('.mol', '.pt') for f in raw_file_names]):
            mol_name = Path(raw_path).name.replace('.mol', '')
            if self.use_cache and self.data_handler.cache and self.data_handler.cache.get(processed_name):
                continue
            mol = self._load_molecule_data(raw_path)
            self._process_molecule_graph(mol, mol_name, processed_name, target_df)
