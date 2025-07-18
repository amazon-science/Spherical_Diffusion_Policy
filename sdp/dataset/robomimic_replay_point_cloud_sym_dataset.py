from typing import Dict, List
import torch
import numpy as np
import h5py
from tqdm import tqdm
import zarr
import os
import shutil
import copy
import json
import hashlib
from filelock import FileLock
from threadpoolctl import threadpool_limits
import concurrent.futures
import multiprocessing
from omegaconf import OmegaConf
from sdp.common.pytorch_util import dict_apply
from sdp.dataset.base_dataset import BaseImageDataset, LinearNormalizer
from sdp.model.common.normalizer import LinearNormalizer, SingleFieldLinearNormalizer
from sdp.model.common.rotation_transformer import RotationTransformer
from sdp.codecs.imagecodecs_numcodecs import register_codecs, Jpeg2k
from sdp.common.replay_buffer import ReplayBuffer
from sdp.common.sampler import SequenceSampler, get_val_mask
from sdp.common.normalize_util import (
    robomimic_abs_action_only_normalizer_from_stat,
    get_range_normalizer_from_stat,
    get_point_cloud_identity_normalizer,
    get_voxel_identity_normalizer,
    get_image_range_normalizer,
    get_identity_normalizer_from_stat,
    array_to_stats,
    get_range_symmetric_normalizer_from_stat
)
register_codecs()

class RobomimicReplayPointCloudSymDataset(BaseImageDataset):
    def __init__(self,
            shape_meta: dict,
            dataset_path: str,
            horizon=1,
            pad_before=0,
            pad_after=0,
            n_obs_steps=None,
            abs_action=False,
            rotation_rep='rotation_6d', # ignored when abs_action=False
            use_legacy_normalizer=False,
            use_cache=False,
            seed=42,
            val_ratio=0.0,
            n_demo=100,
            ws_x_center=0,
            ws_y_center=0,
        ):
        self.n_demo = n_demo
        self.ws_center = np.array([ws_x_center, ws_y_center])
        rotation_transformer = RotationTransformer(
            from_rep='axis_angle', to_rep=rotation_rep)

        replay_buffer = None
        if use_cache:
            cache_zarr_path = dataset_path + f'.{n_demo}.' + '.zarr.zip'
            cache_lock_path = cache_zarr_path + '.lock'
            print('Acquiring lock on cache.')
            with FileLock(cache_lock_path):
                if not os.path.exists(cache_zarr_path):
                    # cache does not exists
                    try:
                        print('Cache does not exist. Creating!')
                        # store = zarr.DirectoryStore(cache_zarr_path)
                        replay_buffer = _convert_point_cloud_to_replay(
                            store=zarr.MemoryStore(), 
                            shape_meta=shape_meta, 
                            dataset_path=dataset_path, 
                            abs_action=abs_action, 
                            rotation_transformer=rotation_transformer,
                            n_demo=n_demo)
                        print('Saving cache to disk.')
                        with zarr.ZipStore(cache_zarr_path) as zip_store:
                            replay_buffer.save_to_store(
                                store=zip_store
                            )
                    except Exception as e:
                        shutil.rmtree(cache_zarr_path)
                        raise e
                else:
                    print('Loading cached ReplayBuffer from Disk.')
                    with zarr.ZipStore(cache_zarr_path, mode='r') as zip_store:
                        replay_buffer = ReplayBuffer.copy_from_store(
                            src_store=zip_store, store=zarr.MemoryStore())
                    print('Loaded!')
        else:
            replay_buffer = _convert_point_cloud_to_replay(
                store=zarr.MemoryStore(), 
                shape_meta=shape_meta, 
                dataset_path=dataset_path, 
                abs_action=abs_action, 
                rotation_transformer=rotation_transformer,
                n_demo=n_demo)
        replay_buffer.data.point_cloud[:, :, :3] -= replay_buffer.data.robot0_eef_pos[:].reshape(-1, 1, 3)
        # replay_buffer.data.action[:, :3] -= replay_buffer.data.robot0_eef_pos[:]
        rgb_keys = list()
        pc_keys = list()
        lowdim_keys = list()
        obs_shape_meta = shape_meta['obs']
        for key, attr in obs_shape_meta.items():
            type = attr.get('type', 'low_dim')
            if type == 'rgb':
                rgb_keys.append(key)
            if type == 'point_cloud':
                pc_keys.append(key)
            elif type == 'low_dim':
                lowdim_keys.append(key)
        
        # for key in rgb_keys:
        #     replay_buffer[key].compressor.numthreads=1

        key_first_k = dict()
        if n_obs_steps is not None:
            # only take first k obs from images
            for key in rgb_keys + pc_keys + lowdim_keys:
                key_first_k[key] = n_obs_steps

        val_mask = get_val_mask(
            n_episodes=replay_buffer.n_episodes, 
            val_ratio=val_ratio,
            seed=seed)
        train_mask = ~val_mask
        sampler = SequenceSampler(
            replay_buffer=replay_buffer, 
            sequence_length=horizon,
            pad_before=pad_before, 
            pad_after=pad_after,
            episode_mask=train_mask,
            key_first_k=key_first_k)
        
        self.replay_buffer = replay_buffer
        self.sampler = sampler
        self.shape_meta = shape_meta
        self.rgb_keys = rgb_keys
        self.pc_keys = pc_keys
        self.lowdim_keys = lowdim_keys
        self.abs_action = abs_action
        self.n_obs_steps = n_obs_steps
        self.train_mask = train_mask
        self.horizon = horizon
        self.pad_before = pad_before
        self.pad_after = pad_after
        self.use_legacy_normalizer = use_legacy_normalizer

    def get_validation_dataset(self):
        val_set = copy.copy(self)
        val_set.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer, 
            sequence_length=self.horizon,
            pad_before=self.pad_before, 
            pad_after=self.pad_after,
            episode_mask=~self.train_mask
            )
        val_set.train_mask = ~self.train_mask
        return val_set

    def get_normalizer(self, mode='limits', **kwargs) -> LinearNormalizer:
        # data = {
        #     'point_cloud': self.replay_buffer['point_cloud'],
        # }
        normalizer = LinearNormalizer()
        stat = array_to_stats(self.replay_buffer['point_cloud'][:].reshape(-1, self.replay_buffer['point_cloud'].shape[-1]))
        xy_magnitute = max(-min(stat['min'][:3]), max(stat['max'][:3]))
        stat['min'][:3] = -xy_magnitute
        stat['max'][:3] = xy_magnitute
        stat['mean'][:3] = 0
        # normalizer['point_cloud'] = get_point_cloud_identity_normalizer(self.replay_buffer['point_cloud'].shape[-1])
        normalizer['point_cloud'] = get_range_symmetric_normalizer_from_stat(stat)
        # normalizer.fit(data=data, last_n_dims=1, mode=mode, **kwargs)
        for key in self.rgb_keys:
            normalizer[key] = get_image_range_normalizer()

        stat = array_to_stats(self.replay_buffer['action'])
        if self.abs_action:
            if stat['mean'].shape[-1] > 10:
                # dual arm
                raise NotImplementedError
            else:
                # magnitute = np.max([stat['max'][:2], -stat['min'][:2]])
                # magnitute = np.linalg.norm(self.replay_buffer['action'][:, :2], axis=1).max()
                max_horizon_mag = 0
                z_min = 10
                z_max = -10
                for i in range(self.replay_buffer.meta.episode_ends.shape[0]):
                    if i == 0:
                        start = 0
                    else:
                        start = self.replay_buffer.meta.episode_ends[i-1]
                    end = self.replay_buffer.meta.episode_ends[i]
                    for horizon in range(self.horizon):
                        max_horizon_steps = self.replay_buffer['action'][start+horizon:end, :3] - self.replay_buffer['robot0_eef_pos'][start:end-horizon]
                        max_horizon_mag = max(max_horizon_mag, np.linalg.norm(np.abs(max_horizon_steps).max(0)[:2]))
                        z_min = min(z_min, max_horizon_steps.min(0)[2])
                        z_max = max(z_max, max_horizon_steps.max(0)[2])

                stat['min'][:2] = -max_horizon_mag
                stat['max'][:2] = max_horizon_mag
                stat['mean'][:2] = 0
                stat['min'][2] = z_min
                stat['max'][2] = z_max
                stat['mean'][2] = np.mean([z_min, z_max])
                this_normalizer = robomimic_abs_action_only_normalizer_from_stat(stat)
            
            if self.use_legacy_normalizer:
                this_normalizer = normalizer_from_stat(stat)
        else:
            # already normalized
            this_normalizer = get_identity_normalizer_from_stat(stat)
        normalizer['action'] = this_normalizer

        # obs
        for key in self.lowdim_keys:
            stat = array_to_stats(self.replay_buffer[key])

            if key.endswith('qpos'):
                this_normalizer = get_range_normalizer_from_stat(stat)
            elif key.endswith('pos'):
                magnitute = np.max([stat['max'][:2] - self.ws_center, self.ws_center - stat['min'][:2]])
                stat['min'][:2] = self.ws_center - magnitute
                stat['max'][:2] = self.ws_center + magnitute
                stat['mean'][:2] = self.ws_center
                this_normalizer = get_range_normalizer_from_stat(stat)
            elif key.endswith('quat'):
                # quaternion is in [-1,1] already
                this_normalizer = get_identity_normalizer_from_stat(stat)
            else:
                raise RuntimeError('unsupported')
            normalizer[key] = this_normalizer

        return normalizer

    def get_all_actions(self) -> torch.Tensor:
        return torch.from_numpy(self.replay_buffer['action'])

    def __len__(self):
        return len(self.sampler)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        threadpool_limits(1)
        data = self.sampler.sample_sequence(idx)

        # to save RAM, only return first n_obs_steps of OBS
        # since the rest will be discarded anyway.
        # when self.n_obs_steps is None
        # this slice does nothing (takes all)
        T_slice = slice(self.n_obs_steps)

        obs_dict = dict()
        for key in self.rgb_keys:
            # move channel last to channel first
            # T,H,W,C
            # convert uint8 image to float32
            obs_dict[key] = np.moveaxis(data[key][T_slice],-1,1
                ).astype(np.float32) / 255.
            # T,C,H,W
            del data[key]
        for key in self.pc_keys:
            obs_dict[key] = data[key][T_slice].astype(np.float32)
            del data[key]
        for key in self.lowdim_keys:
            obs_dict[key] = data[key][T_slice].astype(np.float32)
            del data[key]

        torch_data = {
            'obs': dict_apply(obs_dict, torch.from_numpy),
            'action': torch.from_numpy(data['action'].astype(np.float32))
        }
        return torch_data


def _convert_actions(raw_actions, abs_action, rotation_transformer):
    actions = raw_actions
    if abs_action:
        is_dual_arm = False
        if raw_actions.shape[-1] == 14:
            # dual arm
            raw_actions = raw_actions.reshape(-1,2,7)
            is_dual_arm = True

        pos = raw_actions[...,:3]
        rot = raw_actions[...,3:6]
        gripper = raw_actions[...,6:]
        rot = rotation_transformer.forward(rot)
        raw_actions = np.concatenate([
            pos, rot, gripper
        ], axis=-1).astype(np.float32)
    
        if is_dual_arm:
            raw_actions = raw_actions.reshape(-1,20)
        actions = raw_actions
    return actions


def _convert_point_cloud_to_replay(store, shape_meta, dataset_path, abs_action, rotation_transformer, 
        n_workers=None, max_inflight_tasks=None, n_demo=100):
    if n_workers is None:
        n_workers = 24
    if max_inflight_tasks is None:
        max_inflight_tasks = n_workers * 5

    # parse shape_meta
    pc_keys = list()
    rgb_keys = list()
    lowdim_keys = list()
    # construct compressors and chunks
    obs_shape_meta = shape_meta['obs']
    for key, attr in obs_shape_meta.items():
        shape = attr['shape']
        type = attr.get('type', 'low_dim')
        if type == 'rgb':
            rgb_keys.append(key)
        elif type == 'point_cloud':
            pc_keys.append(key)
        elif type == 'low_dim':
            lowdim_keys.append(key)
    
    root = zarr.group(store)
    data_group = root.require_group('data', overwrite=True)
    meta_group = root.require_group('meta', overwrite=True)

    with h5py.File(dataset_path) as file:
        # count total steps
        demos = file['data']
        episode_ends = list()
        prev_end = 0
        n_demo = min(n_demo, len(demos))
        for i in range(n_demo):
            demo = demos[f'demo_{i}']
            episode_length = demo['actions'].shape[0]
            episode_end = prev_end + episode_length
            prev_end = episode_end
            episode_ends.append(episode_end)
        n_steps = episode_ends[-1]
        episode_starts = [0] + episode_ends[:-1]
        _ = meta_group.array('episode_ends', episode_ends, 
            dtype=np.int64, compressor=None, overwrite=True)

        # save lowdim data
        for key in tqdm(lowdim_keys + ['action'], desc="Loading lowdim data"):
            data_key = 'obs/' + key
            if key == 'action':
                data_key = 'actions'
            this_data = list()
            for i in range(n_demo):
                demo = demos[f'demo_{i}']
                this_data.append(demo[data_key][:].astype(np.float32))
            this_data = np.concatenate(this_data, axis=0)
            if key == 'action':
                this_data = _convert_actions(
                    raw_actions=this_data,
                    abs_action=abs_action,
                    rotation_transformer=rotation_transformer
                )
                assert this_data.shape == (n_steps,) + tuple(shape_meta['action']['shape'])
            else:
                assert this_data.shape == (n_steps,) + tuple(shape_meta['obs'][key]['shape'])
            _ = data_group.array(
                name=key,
                data=this_data,
                shape=this_data.shape,
                chunks=this_data.shape,
                compressor=None,
                dtype=this_data.dtype
            )

        def copy_to_zarr(zarr_arr, hdf5_arr, start_idx, end_idx):
            try:
                zarr_arr[start_idx:end_idx] = hdf5_arr
                # make sure we can successfully decode
                _ = zarr_arr[start_idx:end_idx]
                return True
            except Exception as e:
                return False
        
        def pc_copy(zarr_arr, zarr_idx, hdf5_arr, hdf5_idx):
            try:
                zarr_arr[zarr_idx] = hdf5_arr[hdf5_idx]
                _ = zarr_arr[zarr_idx]
                return True
            except Exception as e:
                return False
        
        def img_copy(zarr_arr, zarr_idx, hdf5_arr, hdf5_idx):
            try:
                zarr_arr[zarr_idx] = hdf5_arr[hdf5_idx]
                # make sure we can successfully decode
                _ = zarr_arr[zarr_idx]
                return True
            except Exception as e:
                return False
            
        with tqdm(total=n_demo*len(rgb_keys), desc="Loading image data", mininterval=1.0) as pbar:
            # one chunk per thread, therefore no synchronization needed
            with concurrent.futures.ThreadPoolExecutor(max_workers=n_workers) as executor:
                futures = set()
                for key in rgb_keys:
                    data_key = 'obs/' + key
                    shape = tuple(shape_meta['obs'][key]['shape'])
                    c,h,w = shape
                    this_compressor = Jpeg2k(level=50)
                    img_arr = data_group.require_dataset(
                        name=key,
                        shape=(n_steps,h,w,c),
                        chunks=(1,h,w,c),
                        compressor=this_compressor,
                        dtype=np.uint8
                    )
                    for episode_idx in range(n_demo):
                        demo = demos[f'demo_{episode_idx}']
                        hdf5_arr = demo['obs'][key][:]
                        start_idx = episode_starts[episode_idx]
                        if episode_idx < n_demo - 1:
                            end_idx = episode_starts[episode_idx+1]
                        else:
                            end_idx = n_steps
                        if len(futures) >= max_inflight_tasks:
                            # limit number of inflight tasks
                            completed, futures = concurrent.futures.wait(futures, 
                                return_when=concurrent.futures.FIRST_COMPLETED)
                            for f in completed:
                                if not f.result():
                                    raise RuntimeError('Failed to encode image!')
                            pbar.update(len(completed))

                        futures.add(
                            executor.submit(copy_to_zarr, 
                                img_arr, hdf5_arr, start_idx, end_idx))
                completed, futures = concurrent.futures.wait(futures)
                for f in completed:
                    if not f.result():
                        raise RuntimeError('Failed to encode image!')
                pbar.update(len(completed))

        with tqdm(total=n_demo*len(pc_keys), desc="Loading point cloud data", mininterval=1.0) as pbar:
            # one chunk per thread, therefore no synchronization needed
            with concurrent.futures.ThreadPoolExecutor(max_workers=n_workers) as executor:
                futures = set()
                for key in pc_keys:
                    data_key = 'obs/' + key
                    shape = tuple(shape_meta['obs'][key]['shape'])
                    n, c = shape
                    img_arr = data_group.require_dataset(
                        name=key,
                        shape=(n_steps, n, c),
                        chunks=(1, n, c),
                        dtype=np.float32
                    )
                    for episode_idx in range(n_demo):
                        demo = demos[f'demo_{episode_idx}']
                        hdf5_arr = demo['obs'][key][:]
                        start_idx = episode_starts[episode_idx]
                        if episode_idx < n_demo - 1:
                            end_idx = episode_starts[episode_idx+1]
                        else:
                            end_idx = n_steps
                        if len(futures) >= max_inflight_tasks:
                            # limit number of inflight tasks
                            completed, futures = concurrent.futures.wait(futures, 
                                return_when=concurrent.futures.FIRST_COMPLETED)
                            for f in completed:
                                if not f.result():
                                    raise RuntimeError('Failed to encode image!')
                            pbar.update(len(completed))

                        futures.add(
                            executor.submit(copy_to_zarr, 
                                img_arr, hdf5_arr, start_idx, end_idx))
                completed, futures = concurrent.futures.wait(futures)
                for f in completed:
                    if not f.result():
                        raise RuntimeError('Failed to encode image!')
                pbar.update(len(completed))

    replay_buffer = ReplayBuffer(root)
    return replay_buffer

def normalizer_from_stat(stat):
    max_abs = np.maximum(stat['max'].max(), np.abs(stat['min']).max())
    scale = np.full_like(stat['max'], fill_value=1/max_abs)
    offset = np.zeros_like(stat['max'])
    return SingleFieldLinearNormalizer.create_manual(
        scale=scale,
        offset=offset,
        input_stats_dict=stat
    )
