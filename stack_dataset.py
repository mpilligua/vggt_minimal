
import os
import json
import torch
import pickle
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
from glob import glob
from tqdm import tqdm
import random


class StackDataset(Dataset):
    def __init__(self, data_dir, 
                 use_cache=True, 
                 cache_path="dataset_gamma_cache_depth.pkl", 
                 transform=None, 
                 experiment_config = None):
        self.data_dir = data_dir
        self.transform = transform
        self.cache_path = os.path.join(data_dir, cache_path)

        if experiment_config is None:
            self.experiment_config = {
                ...
            }
        else:
            self.experiment_config = experiment_config

        # Load cached dataset paths if available
        if use_cache and os.path.exists(self.cache_path):
            print(f"🔹 Loading cached dataset from {self.cache_path}")
            with open(self.cache_path, "rb") as f:
                self.data = pickle.load(f)
        else:
            print(f"⚠️ Cache not found, scanning {data_dir}...")
            self.data = self._build_dataset()
            with open(self.cache_path, "wb") as f:
                pickle.dump(self.data, f)

        # Extract lists for easier access
        self.image_paths = [entry["image_path"] for entry in self.data]
        #self.volume_ratios = [entry["volume_ratio"] for entry in self.data]
        #self.object_volumes = [entry.get("object_volume", 0) for entry in self.data]  # Default to 0 if missing
        self.images = [entry["images"] for entry in self.data]

    def _build_dataset(self):
        """ Scans the dataset folder and builds a list of image paths + volume data. """
        dataset_entries = []

        for folder in tqdm(sorted(os.listdir(self.data_dir))):
            folder_path = os.path.join(self.data_dir, folder)
            if not os.path.isdir(folder_path): #or not folder.endswith("_0"): # TODO option use only 0s
                continue  # Skip if not a valid data folder

            if not(5046 > int(os.path.basename(folder_path).split("_")[0]) > 2217):
                continue

            # Find the first image in alphabetical order
            if self.experiment_config['INPUT_TYPE'] == 'COLOR': # TODO use 'match'
                image_folder = os.path.join(folder_path, "MultiView/RGB")
                image_files = sorted(glob(os.path.join(image_folder, "*")))
                if not image_files:
                    print(f"⚠️ No images found in {image_folder}, skipping...")
                    continue
                image_path = image_files[0]
            elif not self.experiment_config['INPUT_TYPE'] == 'GT_DEPTH':
                image_path = os.path.join(folder_path, "nadir_depth.png")
                if not os.path.exists(image_path):
                    continue

            image_path = os.path.join(folder_path, "images")
            images = sorted(glob(os.path.join(image_path, "*")))

            # Load simulation results JSON
            # json_path = os.path.join(folder_path, "simulation_results.json")
            # if not os.path.exists(json_path):
            #     print(f"⚠️ No JSON found in {folder_path}, skipping...")
            #     continue

            # with open(json_path, "r") as f:
            #     sim_data = json.load(f)

            # Extract volume ratio
            # if self.experiment_config['PREDICT_TYPE'] == 'GAMMA_WITH_EDGES': # TODO use 'match'
            #     volume_ratio = sim_data["volume_ratio_with_edges"]
            # elif self.experiment_config['PREDICT_TYPE'] == 'GAMMA_NO_EDGES':
            #     volume_ratio = sim_data["volume_ratio_no_edges"]
            # elif self.experiment_config['PREDICT_TYPE'] == 'COUNT_DIRECTLY':
            #     volume_ratio = sim_data["object_origins"]

            # Store relevant data
            dataset_entries.append({
                "image_path": image_path,
                # "volume_ratio": volume_ratio,
                "images": images,
                #"object_volume": sim_data.get("object", {}).get("volume", 0)  # Default to 0 if missing
            })

        print(f"✅ Dataset scan complete: {len(dataset_entries)} valid samples found.")
        return dataset_entries

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_paths = self.images[idx]
        image_paths = random.sample(image_paths, min(10, len(image_paths)))
        # volume_ratio = self.volume_ratios[idx]
        # object_volume = self.object_volumes[idx]

        # Preallocate tensor for images
        # Assuming all images are of the same size after transformation
        images = torch.zeros((10, 3, 512, 512), dtype=torch.float32)

        # Load and transform all images at once
        for i, image_path in enumerate(image_paths):
            image = Image.open(image_path).convert("RGB")
            images[i] = self.transform(image)  # Apply the transformation directly

        # Pad the image to be divisible by 14
        pad_h = (14 - images.shape[2] % 14) % 14
        pad_w = (14 - images.shape[3] % 14) % 14
        images = torch.nn.functional.pad(images, (0, pad_w, 0, pad_h))

        # Return data
        # volume_ratio_tensor = torch.tensor(volume_ratio, dtype=torch.float32)
        # object_volume_tensor = torch.tensor(object_volume, dtype=torch.float32)

        folder_path = os.path.basename(os.path.dirname(self.image_paths[idx]))
        image_name = os.path.basename(self.image_paths[idx])

        if self.experiment_config['DINO_USE_VOL_AS_ADDITIONAL_INPUT']:
            return {
                "images":images, 
                # volume_ratio_tensor, 
                # object_volume_tensor
                }
        else:
            return {
                "images": images, 
                # volume_ratio_tensor, 
                "folder": folder_path, 
                "image_name":image_name}


class StackDatasetEmbs(Dataset):
    def __init__(self, data_dir, emb_dir,
                 use_cache=True, 
                 cache_path="dataset_gamma_cache_depth.pkl", 
                 experiment_config = None, 
                 test=False):
        self.data_dir = data_dir
        self.emb_dir = emb_dir
        self.cache_path = os.path.join(data_dir, cache_path)
        self.test = test

        if experiment_config is None:
            self.experiment_config = {
                ...
            }
        else:
            self.experiment_config = experiment_config


        if self.test: 
            with open(os.path.join("/Users/maria/ML/data/test/gt_counts.json"), "r") as f:
                self.counts = json.load(f)

        # Load cached dataset paths if available
        if use_cache and os.path.exists(self.cache_path):
            print(f"🔹 Loading cached dataset from {self.cache_path}")
            with open(self.cache_path, "rb") as f:
                self.data = pickle.load(f)
        else:
            print(f"⚠️ Cache not found, scanning {data_dir}...")
            self.data = self._build_dataset()
            with open(self.cache_path, "wb") as f:
                pickle.dump(self.data, f)

        # Extract lists for easier access
        # self.image_paths = [entry["image_path"] for entry in self.data]
        self.volume_ratios = [entry["volume_ratio"] for entry in self.data]
        # self.object_volumes = [entry.get("object_volume", 0) for entry in self.data]  # Default to 0 if missing
        # self.images = [entry["images"] for entry in self.data]
        #self.embeddings = [entry["embedding"] for entry in self.data]
        # self.folders = sorted(os.listdir(self.data_dir))
        self.folders =  [entry["folder"] for entry in self.data]

    def _build_dataset(self):
        """ Scans the dataset folder and builds a list of image paths + volume data. """
        dataset_entries = []

        for folder in tqdm(sorted(os.listdir(self.emb_dir))):
            folder_path = os.path.join(self.data_dir, folder)
            if not os.path.isdir(folder_path): #or not folder.endswith("_0"): # TODO option use only 0s
                continue  # Skip if not a valid data folder

            # image_path = os.path.join(folder_path, "images")
            # images = sorted(glob(os.path.join(image_path, "*")))

            # scene_name = os.path.basename(folder_path)
            # embedding = torch.load(self.data_dir + "/"+ folder + "/images.pt")

            # Load simulation results JSON
            if not self.test: 
                json_path = os.path.join(folder_path, "simulation_results.json")
                if not os.path.exists(json_path):
                    print(f"⚠️ No JSON found in {folder_path}, skipping...")
                    continue

                with open(json_path, "r") as f:
                    sim_data = json.load(f)

                # print(sim_data)

                # Extract volume ratio
                # print("Predicting", self.experiment_config['PREDICT_TYPE'])
                if self.experiment_config['PREDICT_TYPE'] == 'GAMMA_WITH_EDGES': # TODO use 'match'
                    volume_ratio = sim_data["volume_ratio_with_edges"]
                elif self.experiment_config['PREDICT_TYPE'] == 'GAMMA_NO_EDGES':
                    volume_ratio = sim_data["volume_ratio_no_edges"]
                elif self.experiment_config['PREDICT_TYPE'] == 'COUNT_DIRECTLY':
                    volume_ratio = sim_data["obj_origins"]

            else: 
                volume_ratio = self.counts[folder]

            # Store relevant data
            dataset_entries.append({
                "folder": folder,
                # "image_path": image_path,
                "volume_ratio": volume_ratio,
                # "images": images,
                # "embedding": embedding,
                #"object_volume": sim_data.get("object", {}).get("volume", 0)  # Default to 0 if missing
            })

        print(f"✅ Dataset scan complete: {len(dataset_entries)} valid samples found.")
        return dataset_entries

    def __len__(self):
        return len(self.folders)

    def __getitem__(self, idx):
        # image_paths = self.images[idx]
        volume_ratio = self.volume_ratios[idx]
        # object_volume = self.object_volumes[idx]
        folder = self.folders[idx]
        try: 
            embedding = torch.load(self.emb_dir + "/"+ folder + "/images.pt")
        except:
            embedding = torch.load(self.emb_dir + "/"+ folder + "/images_mean.pt")

        
        # Preallocate tensor for images
        # Assuming all images are of the same size after transformation
        # images = torch.zeros((len(image_paths), 3, 512, 512), dtype=torch.float32)

        # Load and transform all images at once
        # for i, image_path in enumerate(image_paths):
            # image = Image.open(image_path).convert("RGB")
            # images[i] = self.transform(image)  # Apply the transformation directly

        # Pad the image to be divisible by 14
        # pad_h = (14 - images.shape[2] % 14) % 14
        # pad_w = (14 - images.shape[3] % 14) % 14
        # images = torch.nn.functional.pad(images, (0, pad_w, 0, pad_h))

        # Return data
        volume_ratio_tensor = torch.tensor(volume_ratio, dtype=torch.float32)
        # object_volume_tensor = torch.tensor(object_volume, dtype=torch.float32)
        # embedding = torch.tensor(embedding, dtype=torch.float32)

        # folder_path = os.path.basename(os.path.dirname(self.image_paths[idx]))
        # image_name = os.path.basename(self.image_paths[idx])


        return {
            "embedding": embedding,
            "volume_ratio": volume_ratio_tensor,
            # "object_volume": object_volume_tensor,
            # "folder_path": folder_path,
            # "image_name": image_name
        }




class StackDataset2(Dataset):
    def __init__(self, data_dir,
                 use_cache=True, 
                 num_views=3,
                 force_top_image=False,
                 cache_path="dataset_gamma_cache_depth_all.pkl", 
                 experiment_config = None, 
                 test=False,
                 transform=None):
        self.data_dir = data_dir
        self.cache_path = os.path.join(data_dir, cache_path)
        self.test = test
        self.transform = transform

        if experiment_config is None:
            self.experiment_config = {
                ...
            }
        else:
            self.experiment_config = experiment_config


        if self.test: 
            with open(os.path.join("/Users/maria/ML/data/test/gt_counts.json"), "r") as f:
                self.counts = json.load(f)

        # Load cached dataset paths if available
        if use_cache and os.path.exists(self.cache_path):
            print(f"🔹 Loading cached dataset from {self.cache_path}")
            with open(self.cache_path, "rb") as f:
                self.data = pickle.load(f)
        else:
            print(f"⚠️ Cache not found, scanning {data_dir}...")
            self.data = self._build_dataset()
            with open(self.cache_path, "wb") as f:
                pickle.dump(self.data, f)

        # Extract lists for easier access
        self.image_paths = [entry["image_path"] for entry in self.data]
        self.volume_ratios = [entry["volume_ratio"] for entry in self.data]
        self.images = [entry["images"] for entry in self.data]
        self.folders =  [entry["folder"] for entry in self.data]
        self.num_views = num_views
        self.force_top_image = force_top_image

    def _build_dataset(self):
        """ Scans the dataset folder and builds a list of image paths + volume data. """
        dataset_entries = []

        for folder in tqdm(sorted(os.listdir(self.data_dir))):
            folder_path = os.path.join(self.data_dir, folder)
            if not os.path.isdir(folder_path): #or not folder.endswith("_0"): # TODO option use only 0s
                continue  # Skip if not a valid data folder

            image_path = os.path.join(folder_path, "images")
            images = sorted(glob(os.path.join(image_path, "*")))

            # scene_name = os.path.basename(folder_path)
            # embedding = torch.load(self.data_dir + "/"+ folder + "/images.pt")

            # Load simulation results JSON
            if not self.test: 
                json_path = os.path.join(folder_path, "simulation_results.json")
                if not os.path.exists(json_path):
                    print(f"⚠️ No JSON found in {folder_path}, skipping...")
                    continue

                with open(json_path, "r") as f:
                    sim_data = json.load(f)

                # print(sim_data)

                # Extract volume ratio
                # print("Predicting", self.experiment_config['PREDICT_TYPE'])
                if self.experiment_config['PREDICT_TYPE'] == 'GAMMA_WITH_EDGES': # TODO use 'match'
                    volume_ratio = sim_data["volume_ratio_with_edges"]
                elif self.experiment_config['PREDICT_TYPE'] == 'GAMMA_NO_EDGES':
                    volume_ratio = sim_data["volume_ratio_no_edges"]
                elif self.experiment_config['PREDICT_TYPE'] == 'COUNT_DIRECTLY':
                    volume_ratio = sim_data["obj_origins"]

            else: 
                volume_ratio = self.counts[folder]

            # Store relevant data
            dataset_entries.append({
                "folder": folder,
                "image_path": image_path,
                "volume_ratio": volume_ratio,
                "images": images,
                # "embedding": embedding,
                #"object_volume": sim_data.get("object", {}).get("volume", 0)  # Default to 0 if missing
            })

        print(f"✅ Dataset scan complete: {len(dataset_entries)} valid samples found.")
        return dataset_entries

    def __len__(self):
        return len(self.folders)

    def __getitem__(self, idx):
        image_paths = self.images[idx]
        if self.force_top_image:
            image_paths = random.sample(image_paths[1:], min(self.num_views-1, len(image_paths)-1))
            image_paths.insert(0, sorted(self.images[idx])[0])  # Always include the top image
        else:
            image_paths = random.sample(image_paths, min(self.num_views, len(image_paths)))
        volume_ratio = self.volume_ratios[idx]
        # object_volume = self.object_volumes[idx]
        folder = self.folders[idx]
        
        # Preallocate tensor for images
        # Assuming all images are of the same size after transformation
        images = torch.zeros((len(image_paths), 3, 518, 518), dtype=torch.float32)

        # Load and transform all images at once
        for i, image_path in enumerate(image_paths):
            image = Image.open(image_path).convert("RGB")
            images[i] = self.transform(image)  # Apply the transformation directly

        # Pad the image to be divisible by 14
        pad_h = (14 - images.shape[2] % 14) % 14
        pad_w = (14 - images.shape[3] % 14) % 14
        images = torch.nn.functional.pad(images, (0, pad_w, 0, pad_h))

        # Return data
        volume_ratio_tensor = torch.tensor(volume_ratio, dtype=torch.float32)
        # object_volume_tensor = torch.tensor(object_volume, dtype=torch.float32)
        # embedding = torch.tensor(embedding, dtype=torch.float32)

        # folder_path = os.path.basename(os.path.dirname(self.image_paths[idx]))
        image_name = os.path.basename(self.image_paths[idx])


        return {
            "images": images,
            "volume_ratio": volume_ratio_tensor,
            # "object_volume": object_volume_tensor,
            "folder": folder,
            "image_name": image_name
        }
