# --- dataset.py ---
import torch
import torch_geometric.data
import torch_geometric.transforms
import pandas as pd
from pathlib import Path
from data_utils import DataProcessor, DatasetError, EmptyDatasetError
from config_loader import Config
import logging

logger = logging.getLogger(__name__)

class MGDataset(torch_geometric.data.Dataset):
    def __init__(self, root, directed=False, transform=None, target_csv='targets_g_reg.csv', use_cache=True, rdkit_config: Config = None):
        self.directed = directed
        try:
            self.target_df = pd.read_csv(Path(root) / target_csv, index_col=['MoleculeName', ])
            if self.target_df.empty:
                raise EmptyDatasetError("Target CSV file is empty.")
        except FileNotFoundError:
            raise DatasetError(f"Target CSV file not found: {target_csv}")
        except Exception as e:
            raise DatasetError(f"Error loading target CSV: {e}")
        self.use_cache = use_cache
        self.root = root
        self.data_processor = DataProcessor(root, use_cache, rdkit_config)
        self.pre_transform = torch_geometric.transforms.Compose([torch_geometric.transforms.NormalizeFeatures(), torch_geometric.transforms.AddSelfLoops(), torch_geometric.transforms.Distance()])
        super().__init__(root, transform)
        self.transform_list = torch_geometric.transforms.Compose([torch_geometric.transforms.RandomRotate(degrees=180), torch_geometric.transforms.RandomScale((0.9, 1.1)), torch_geometric.transforms.RandomJitter(0.01), torch_geometric.transforms.RandomFlip(0), ])
        self.data_processor.process_data(self.target_df)
        self.data_list = self._load_and_transform_data()

    def _load_and_transform_data(self):
        data_list = []
        for processed_file_name in self.processed_file_names:
            data = self.data_processor.data_handler.load_graph(processed_file_name)
            if data is not None:
                data = self.transform_list(data)
                data_list.append(data)
            else:
                logger.warning(f"Skipping molecule {processed_file_name} due to missing data.")
        return data_list

    @property
    def raw_file_names(self):
        return self.data_processor.raw_loader.load_raw_files()

    @property
    def processed_file_names(self):
        return [Path(f).name.replace('.mol', '.pt') for f in self.raw_file_names]

    def download(self):
        pass

    def process(self):
        pass

    def len(self):
        return len(self.data_list)

    def get(self, idx):
        return self.data_list[idx]

    def get_molecule_name(self, idx: int) -> str:
        processed_file_name = self.processed_file_names[idx]
        return Path(processed_file_name).stem
