import abc
import copy
import json
import os
import cv2
from internal import camera_utils
from internal import configs
from internal import image as lib_image
from internal import raw_utils
from internal import utils
from collections import defaultdict
import numpy as np
import cv2
from PIL import Image
import torch
from tqdm import tqdm
# This is ugly, but it works.
import sys

sys.path.insert(0, 'internal/pycolmap')
sys.path.insert(0, 'internal/pycolmap/pycolmap')
import pycolmap


def load_dataset(split, train_dir, config: configs.Config):
    """Loads a split of a dataset using the data_loader specified by `config`."""
    if config.multiscale:
        dataset_dict = {
            'llff': MultiLLFF,
        }
    else:
        dataset_dict = {
            'blender': Blender,
            'llff': LLFF,
            'tat_nerfpp': TanksAndTemplesNerfPP,
            'tat_fvs': TanksAndTemplesFVS,
            'dtu': DTU,
        }
    return dataset_dict[config.dataset_loader](split, train_dir, config)


class NeRFSceneManager(pycolmap.SceneManager):
    """COLMAP pose loader.

    Minor NeRF-specific extension to the third_party Python COLMAP loader:
    google3/third_party/py/pycolmap/scene_manager.py
    """

    def process(self):
        """Applies NeRF-specific postprocessing to the loaded pose data.

        Returns:
        a tuple [image_names, poses, pixtocam, distortion_params].
        image_names:  contains the only the basename of the images.
        poses: [N, 4, 4] array containing the camera to world matrices.
        pixtocam: [N, 3, 3] array containing the camera to pixel space matrices.
        distortion_params: mapping of distortion param name to distortion
            parameters. Cameras share intrinsics. Valid keys are k1, k2, p1 and p2.
        """

        self.load_cameras()
        self.load_images()
        # self.load_points3D()  # For now, we do not need the point cloud data.

        # Assume shared intrinsics between all cameras.
        cam = self.cameras[1]

        # Extract focal lengths and principal point parameters.
        fx, fy, cx, cy = cam.fx, cam.fy, cam.cx, cam.cy
        pixtocam = np.linalg.inv(camera_utils.intrinsic_matrix(fx, fy, cx, cy))

        # Extract extrinsic matrices in world-to-camera format.
        imdata = self.images
        w2c_mats = []
        bottom = np.array([0, 0, 0, 1]).reshape(1, 4)
        for k in imdata:
            im = imdata[k]
            rot = im.R()
            trans = im.tvec.reshape(3, 1)
            w2c = np.concatenate([np.concatenate([rot, trans], 1), bottom], axis=0)
            w2c_mats.append(w2c)
        w2c_mats = np.stack(w2c_mats, axis=0)

        # Convert extrinsics to camera-to-world.
        c2w_mats = np.linalg.inv(w2c_mats)
        poses = c2w_mats[:, :3, :4]

        # Image names from COLMAP. No need for permuting the poses according to
        # image names anymore.
        names = [imdata[k].name for k in imdata]

        # Switch from COLMAP (right, down, fwd) to NeRF (right, up, back) frame.
        poses = poses @ np.diag([1, -1, -1, 1])

        # Get distortion parameters.
        type_ = cam.camera_type

        if type_ == 0 or type_ == 'SIMPLE_PINHOLE':
            params = None
            camtype = camera_utils.ProjectionType.PERSPECTIVE

        elif type_ == 1 or type_ == 'PINHOLE':
            params = None
            camtype = camera_utils.ProjectionType.PERSPECTIVE

        if type_ == 2 or type_ == 'SIMPLE_RADIAL':
            params = {k: 0. for k in ['k1', 'k2', 'k3', 'p1', 'p2']}
            params['k1'] = cam.k1
            camtype = camera_utils.ProjectionType.PERSPECTIVE

        elif type_ == 3 or type_ == 'RADIAL':
            params = {k: 0. for k in ['k1', 'k2', 'k3', 'p1', 'p2']}
            params['k1'] = cam.k1
            params['k2'] = cam.k2
            camtype = camera_utils.ProjectionType.PERSPECTIVE

        elif type_ == 4 or type_ == 'OPENCV':
            params = {k: 0. for k in ['k1', 'k2', 'k3', 'p1', 'p2']}
            params['k1'] = cam.k1
            params['k2'] = cam.k2
            params['p1'] = cam.p1
            params['p2'] = cam.p2
            camtype = camera_utils.ProjectionType.PERSPECTIVE

        elif type_ == 5 or type_ == 'OPENCV_FISHEYE':
            params = {k: 0. for k in ['k1', 'k2', 'k3', 'k4']}
            params['k1'] = cam.k1
            params['k2'] = cam.k2
            params['k3'] = cam.k3
            params['k4'] = cam.k4
            camtype = camera_utils.ProjectionType.FISHEYE

        return names, poses, pixtocam, params, camtype


def load_blender_posedata(data_dir, split=None):
    """Load poses from `transforms.json` file, as used in Blender/NGP datasets."""
    suffix = '' if split is None else f'_{split}'
    pose_file = os.path.join(data_dir, f'transforms{suffix}.json')
    with utils.open_file(pose_file, 'r') as fp:
        meta = json.load(fp)
    names = []
    poses = []
    for _, frame in enumerate(meta['frames']):
        filepath = os.path.join(data_dir, frame['file_path'])
        if utils.file_exists(filepath):
            names.append(frame['file_path'].split('/')[-1])
            poses.append(np.array(frame['transform_matrix'], dtype=np.float32))
    poses = np.stack(poses, axis=0)

    w = meta['w']
    h = meta['h']
    cx = meta['cx'] if 'cx' in meta else w / 2.
    cy = meta['cy'] if 'cy' in meta else h / 2.
    if 'fl_x' in meta:
        fx = meta['fl_x']
    else:
        fx = 0.5 * w / np.tan(0.5 * float(meta['camera_angle_x']))
    if 'fl_y' in meta:
        fy = meta['fl_y']
    else:
        fy = 0.5 * h / np.tan(0.5 * float(meta['camera_angle_y']))
    pixtocam = np.linalg.inv(camera_utils.intrinsic_matrix(fx, fy, cx, cy))
    coeffs = ['k1', 'k2', 'p1', 'p2']
    if not any([c in meta for c in coeffs]):
        params = None
    else:
        params = {c: (meta[c] if c in meta else 0.) for c in coeffs}
    camtype = camera_utils.ProjectionType.PERSPECTIVE
    return names, poses, pixtocam, params, camtype


class Dataset(torch.utils.data.Dataset):
    """Dataset Base Class.

    Base class for a NeRF dataset. Creates batches of ray and color data used for
    training or rendering a NeRF model.

    Each subclass is responsible for loading images and camera poses from disk by
    implementing the _load_renderings() method. This data is used to generate
    train and test batches of ray + color data for feeding through the NeRF model.
    The ray parameters are calculated in _generate_rays().

    The public interface mimics the behavior of a standard machine learning
    pipeline dataset provider that can provide infinite batches of data to the
    training/testing pipelines without exposing any details of how the batches are
    loaded/created or how this is parallelized. Therefore, the initializer runs
    all setup, including data loading from disk using _load_renderings(), and
    begins the thread using its parent start() method. After the initializer
    returns, the caller can request batches of data straight away.

    The internal self._queue is initialized as queue.Queue(3), so the infinite
    loop in run() will block on the call self._queue.put(self._next_fn()) once
    there are 3 elements. The main thread training job runs in a loop that pops 1
    element at a time off the front of the queue. The Dataset thread's run() loop
    will populate the queue with 3 elements, then wait until a batch has been
    removed and push one more onto the end.

    This repeats indefinitely until the main thread's training loop completes
    (typically hundreds of thousands of iterations), then the main thread will
    exit and the Dataset thread will automatically be killed since it is a daemon.

    Attributes:
        alphas: np.ndarray, optional array of alpha channel data.
        cameras: tuple summarizing all camera extrinsic/intrinsic/distortion params.
        camtoworlds: np.ndarray, a list of extrinsic camera pose matrices.
        camtype: camera_utils.ProjectionType, fisheye or perspective camera.
        data_dir: str, location of the dataset on disk.
        disp_images: np.ndarray, optional array of disparity (inverse depth) data.
        distortion_params: dict, the camera distortion model parameters.
        exposures: optional per-image exposure value (shutter * ISO / 1000).
        far: float, far plane value for rays.
        focal: float, focal length from camera intrinsics.
        height: int, height of images.
        images: np.ndarray, array of RGB image data.
        metadata: dict, optional metadata for raw datasets.
        near: float, near plane value for rays.
        normal_images: np.ndarray, optional array of surface normal vector data.
        pixtocams: np.ndarray, one or a list of inverse intrinsic camera matrices.
        pixtocam_ndc: np.ndarray, the inverse intrinsic matrix used for NDC space.
        poses: np.ndarray, optional array of auxiliary camera pose data.
        rays: utils.Rays, ray data for every pixel in the dataset.
        render_exposures: optional list of exposure values for the render path.
        render_path: bool, indicates if a smooth camera path should be generated.
        size: int, number of images in the dataset.
        split: str, indicates if this is a "train" or "test" dataset.
        width: int, width of images.
    """

    def __init__(self,
                 split: str,
                 data_dir: str,
                 config: configs.Config):
        super().__init__()

        # Initialize attributes
        self._patch_size = max(config.patch_size, 1)
        self._batch_size = config.batch_size // config.world_size
        if self._patch_size ** 2 > self._batch_size:
            raise ValueError(f'Patch size {self._patch_size}^2 too large for ' +
                             f'per-process batch size {self._batch_size}')
        self._batching = utils.BatchingMethod(config.batching)
        self._uniform_batching_from_all_views = config.uniform_batching_from_all_views
        self._use_tiffs = config.use_tiffs
        self._load_disps = config.compute_disp_metrics
        self._load_normals = config.compute_normal_metrics
        self._num_border_pixels_to_mask = config.num_border_pixels_to_mask
        self._apply_bayer_mask = config.apply_bayer_mask
        self._render_spherical = False

        self.config = config
        self.global_rank = config.global_rank
        self.world_size = config.world_size
        self.split = utils.DataSplit(split)
        self.data_dir = data_dir
        self.near = config.near
        self.far = config.far
        self.render_path = config.render_path
        self.distortion_params = None
        self.disp_images = None
        self.normal_images = None
        self.alphas = None
        self.poses = None
        self.pixtocam_ndc = None
        self.metadata = None
        self.camtype = camera_utils.ProjectionType.PERSPECTIVE
        self.exposures = None
        self.render_exposures = None

        # Providing type comments for these attributes, they must be correctly
        # initialized by _load_renderings() (see docstring) in any subclass.
        self.images: np.ndarray = None
        self.camtoworlds: np.ndarray = None
        self.pixtocams: np.ndarray = None
        self.height: int = None
        self.width: int = None

        # Load data from disk using provided config parameters.
        self._load_renderings(config)

        if self.render_path:
            if config.render_path_file is not None:
                with utils.open_file(config.render_path_file, 'rb') as fp:
                    render_poses = np.load(fp)
                self.camtoworlds = render_poses
            if config.render_resolution is not None:
                self.width, self.height = config.render_resolution
            if config.render_focal is not None:
                self.focal = config.render_focal
            if config.render_camtype is not None:
                if config.render_camtype == 'pano':
                    self._render_spherical = True
                else:
                    self.camtype = camera_utils.ProjectionType(config.render_camtype)

            self.distortion_params = None
            self.pixtocams = camera_utils.get_pixtocam(self.focal, self.width,
                                                       self.height)

        self._n_examples = self.camtoworlds.shape[0]
        if self._uniform_batching_from_all_views and self._patch_size ** 2 * self._n_examples > self._batch_size:
            raise ValueError(f'Batch size {self._batch_size} too small for ' +
                             f'batching uniformly from all views')

        self.cameras = (self.pixtocams,
                        self.camtoworlds,
                        self.distortion_params,
                        self.pixtocam_ndc)

        # Seed the queue with one batch to avoid race condition.
        if self.split == utils.DataSplit.TRAIN and not config.compute_visibility:
            self._next_fn = self._next_train
        else:
            self._next_fn = self._next_test
        
    @property
    def size(self):
        return self._n_examples

    def __len__(self):
        if self.split == utils.DataSplit.TRAIN and not self.config.compute_visibility:
            return 1000
        else:
            return self._n_examples

    @abc.abstractmethod
    def _load_renderings(self, config):
        """Load images and poses from disk.

        Args:
        config: utils.Config, user-specified config parameters.
        In inherited classes, this method must set the following public attributes:
        images: [N, height, width, 3] array for RGB images.
        disp_images: [N, height, width] array for depth data (optional).
        normal_images: [N, height, width, 3] array for normals (optional).
        camtoworlds: [N, 3, 4] array of extrinsic pose matrices.
        poses: [..., 3, 4] array of auxiliary pose data (optional).
        pixtocams: [N, 3, 4] array of inverse intrinsic matrices.
        distortion_params: dict, camera lens distortion model parameters.
        height: int, height of images.
        width: int, width of images.
        focal: float, focal length to use for ideal pinhole rendering.
        """

    def _make_ray_batch(self,
                        pix_x_int,
                        pix_y_int,
                        cam_idx,
                        lossmult=None
                        ):
        """Creates ray data batch from pixel coordinates and camera indices.

        All arguments must have broadcastable shapes. If the arguments together
        broadcast to a shape [a, b, c, ..., z] then the returned utils.Rays object
        will have array attributes with shape [a, b, c, ..., z, N], where N=3 for
        3D vectors and N=1 for per-ray scalar attributes.

        Args:
        pix_x_int: int array, x coordinates of image pixels.
        pix_y_int: int array, y coordinates of image pixels.
        cam_idx: int or int array, camera indices.
        lossmult: float array, weight to apply to each ray when computing loss fn.

        Returns:
        A dict mapping from strings utils.Rays or arrays of image data.
        This is the batch provided for one NeRF train or test iteration.
        """

        broadcast_scalar = lambda x: np.broadcast_to(x, pix_x_int.shape)[..., None]
        ray_kwargs = {
            'lossmult': broadcast_scalar(1.) if lossmult is None else lossmult,
            'near': broadcast_scalar(self.near),
            'far': broadcast_scalar(self.far),
            'cam_idx': broadcast_scalar(cam_idx),
        }
        # Collect per-camera information needed for each ray.
        if self.metadata is not None:
            # Exposure index and relative shutter speed, needed for RawNeRF.
            for key in ['exposure_idx', 'exposure_values']:
                idx = 0 if self.render_path else cam_idx
                ray_kwargs[key] = broadcast_scalar(self.metadata[key][idx])
        if self.exposures is not None:
            idx = 0 if self.render_path else cam_idx
            ray_kwargs['exposure_values'] = broadcast_scalar(self.exposures[idx])
        if self.render_path and self.render_exposures is not None:
            ray_kwargs['exposure_values'] = broadcast_scalar(
                self.render_exposures[cam_idx])

        pixels = dict(pix_x_int=pix_x_int, pix_y_int=pix_y_int, **ray_kwargs)

        # Slow path, do ray computation using numpy (on CPU).
        batch = camera_utils.cast_ray_batch(self.cameras, pixels, self.camtype)
        batch['cam_dirs'] = -self.camtoworlds[ray_kwargs['cam_idx'][..., 0]][..., :3, 2]

        # import trimesh
        # pts = batch['origins'][..., None, :] + batch['directions'][..., None, :] * np.linspace(0, 1, 5)[:, None]
        # trimesh.Trimesh(vertices=pts.reshape(-1, 3)).export("test.ply", "ply")
        #
        # pts = batch['origins'][0, 0, None, :] - self.camtoworlds[cam_idx][:, 2] * np.linspace(0, 1, 100)[:, None]
        # trimesh.Trimesh(vertices=pts.reshape(-1, 3)).export("test2.ply", "ply")

        if not self.render_path:
            batch['rgb'] = self.images[cam_idx, pix_y_int, pix_x_int]
        if self._load_disps:
            batch['disps'] = self.disp_images[cam_idx, pix_y_int, pix_x_int]
        if self._load_normals:
            batch['normals'] = self.normal_images[cam_idx, pix_y_int, pix_x_int]
            batch['alphas'] = self.alphas[cam_idx, pix_y_int, pix_x_int]
        pix_x_float = (pix_x_int.astype(np.float32) + 0.5) / self.width
        pix_y_float = (pix_y_int.astype(np.float32) + 0.5) / self.height
        batch['pix_xy'] = np.stack([pix_x_float, pix_y_float], axis=-1)
        return {k: torch.from_numpy(v.copy()).float() if v is not None else None for k, v in batch.items()}

    def _next_train(self, item):
        """Sample next training batch (random rays)."""
        # We assume all images in the dataset are the same resolution, so we can use
        # the same width/height for sampling all pixels coordinates in the batch.
        # Batch/patch sampling parameters.
        num_patches = self._batch_size // self._patch_size ** 2
        if self._uniform_batching_from_all_views:
            num_patches = (num_patches // self._n_examples) * self._n_examples
        lower_border = self._num_border_pixels_to_mask
        upper_border = self._num_border_pixels_to_mask + self._patch_size - 1
        # Random pixel patch x-coordinates.
        pix_x_int = np.random.randint(lower_border, self.width - upper_border,
                                      (num_patches, 1, 1))
        # Random pixel patch y-coordinates.
        pix_y_int = np.random.randint(lower_border, self.height - upper_border,
                                      (num_patches, 1, 1))
        # Add patch coordinate offsets.
        # Shape will broadcast to (num_patches, _patch_size, _patch_size).
        patch_dx_int, patch_dy_int = camera_utils.pixel_coordinates(
            self._patch_size, self._patch_size)
        pix_x_int = pix_x_int + patch_dx_int
        pix_y_int = pix_y_int + patch_dy_int
        # Random camera indices.
        if self._batching == utils.BatchingMethod.ALL_IMAGES:
            if self._uniform_batching_from_all_views:
                cam_idx = np.repeat(np.arange(0, self._n_examples), num_patches // self._n_examples)
                cam_idx = cam_idx[:, None, None]
            else:
                cam_idx = np.random.randint(0, self._n_examples, (num_patches, 1, 1))
        else:
            cam_idx = np.random.randint(0, self._n_examples, (1,))

        if self._apply_bayer_mask:
            # Compute the Bayer mosaic mask for each pixel in the batch.
            lossmult = raw_utils.pixels_to_bayer_mask(pix_x_int, pix_y_int)
        else:
            lossmult = None

        return self._make_ray_batch(pix_x_int, pix_y_int, cam_idx,
                                    lossmult=lossmult)

    def generate_ray_batch(self, cam_idx: int):
        """Generate ray batch for a specified camera in the dataset."""
        if self._render_spherical:
            camtoworld = self.camtoworlds[cam_idx]
            rays = camera_utils.cast_spherical_rays(
                camtoworld, self.height, self.width, self.near, self.far)
            return rays
        else:
            # Generate rays for all pixels in the image.
            pix_x_int, pix_y_int = camera_utils.pixel_coordinates(
                self.width, self.height)  # (height, width)
            return self._make_ray_batch(pix_x_int, pix_y_int, cam_idx)

    def _next_test(self, item):
        """Sample next test batch (one full image)."""
        return self.generate_ray_batch(item)

    def collate_fn(self, item):
        return self._next_fn(item[0])

    def __getitem__(self, item):
        return self._next_fn(item)


class Blender(Dataset):
    """Blender Dataset."""

    def _load_renderings(self, config):
        """Load images from disk."""
        if config.render_path:
            raise ValueError('render_path cannot be used for the blender dataset.')
        pose_file = os.path.join(self.data_dir, f'transforms_{self.split.value}.json')
        with utils.open_file(pose_file, 'r') as fp:
            meta = json.load(fp)
        images = []
        disp_images = []
        normal_images = []
        cams = []
        for idx, frame in enumerate(tqdm(meta['frames'], desc='Loading Blender dataset', disable=self.global_rank != 0, leave=False)):
            fprefix = os.path.join(self.data_dir, frame['file_path'])

            def get_img(f, fprefix=fprefix):
                image = utils.load_img(fprefix + f)
                if config.factor > 1:
                    image = lib_image.downsample(image, config.factor)
                return image

            if self._use_tiffs:
                channels = [get_img(f'_{ch}.tiff') for ch in ['R', 'G', 'B', 'A']]
                # Convert image to sRGB color space.
                image = lib_image.linear_to_srgb_np(np.stack(channels, axis=-1))
            else:
                image = get_img('.png') / 255.
            images.append(image)

            if self._load_disps:
                disp_image = get_img('_disp.tiff')
                disp_images.append(disp_image)
            if self._load_normals:
                normal_image = get_img('_normal.png')[..., :3] * 2. / 255. - 1.
                normal_images.append(normal_image)

            cams.append(np.array(frame['transform_matrix'], dtype=np.float32))

        self.images = np.stack(images, axis=0)
        if self._load_disps:
            self.disp_images = np.stack(disp_images, axis=0)
        if self._load_normals:
            self.normal_images = np.stack(normal_images, axis=0)
            self.alphas = self.images[..., -1]

        rgb, alpha = self.images[..., :3], self.images[..., -1:]
        self.images = rgb * alpha + (1. - alpha)  # Use a white background.
        self.height, self.width = self.images.shape[1:3]
        self.camtoworlds = np.stack(cams, axis=0)
        self.focal = .5 * self.width / np.tan(.5 * float(meta['camera_angle_x']))
        self.pixtocams = camera_utils.get_pixtocam(self.focal, self.width,
                                                   self.height)


class LLFF(Dataset):
    """LLFF Dataset."""

    def _load_renderings(self, config):
        """Load images from disk."""
        # Set up scaling factor.
        image_dir_suffix = ''
        # Use downsampling factor (unless loading training split for raw dataset,
        # we train raw at full resolution because of the Bayer mosaic pattern).
        if config.factor > 0 and not (config.rawnerf_mode and
                                      self.split == utils.DataSplit.TRAIN):
            image_dir_suffix = f'_{config.factor}'
            factor = config.factor
        else:
            factor = 1

        # Copy COLMAP data to local disk for faster loading.
        colmap_dir = os.path.join(self.data_dir, 'sparse/0/')

        # Load poses.
        if utils.file_exists(colmap_dir):
            pose_data = NeRFSceneManager(colmap_dir).process()
        else:
            # # Attempt to load Blender/NGP format if COLMAP data not present.
            # pose_data = load_blender_posedata(self.data_dir)
            raise ValueError('COLMAP data not found.')
        image_names, poses, pixtocam, distortion_params, camtype = pose_data

        # Previous NeRF results were generated with images sorted by filename,
        # use this flag to ensure metrics are reported on the same test set.
        inds = np.argsort(image_names)
        image_names = [image_names[i] for i in inds]
        poses = poses[inds]

        # Load bounds if possible (only used in forward facing scenes).
        posefile = os.path.join(self.data_dir, 'poses_bounds.npy')
        if utils.file_exists(posefile):
            with utils.open_file(posefile, 'rb') as fp:
                poses_arr = np.load(fp)
            bounds = poses_arr[:, -2:]
        else:
            bounds = np.array([0.01, 1.])
        self.colmap_to_world_transform = np.eye(4)

        # Scale the inverse intrinsics matrix by the image downsampling factor.
        pixtocam = pixtocam @ np.diag([factor, factor, 1.])
        self.pixtocams = pixtocam.astype(np.float32)
        self.focal = 1. / self.pixtocams[0, 0]
        self.distortion_params = distortion_params
        self.camtype = camtype

        # Load extended metadata
        extconf_file = os.path.join(self.data_dir, 'ext_metadata.json')
        if utils.file_exists(extconf_file):
            with open(extconf_file, 'r') as extconf_fp:
                extconf = json.loads(extconf_fp.read())
        else:
            extconf = dict()

        # Separate out 360 versus forward facing scenes.
        if config.forward_facing:
            # Set the projective matrix defining the NDC transformation.
            self.pixtocam_ndc = self.pixtocams.reshape(-1, 3, 3)[0]
            # Rescale according to a default bd factor.
            scale = 1. / (bounds.min() * .75)
            poses[:, :3, 3] *= scale
            self.colmap_to_world_transform = np.diag([scale] * 3 + [1])
            bounds *= scale
            # Recenter poses.
            poses, transform = camera_utils.recenter_poses(poses)
            self.colmap_to_world_transform = (
                    transform @ self.colmap_to_world_transform)
            # Forward-facing spiral render path.
            self.render_poses = camera_utils.generate_spiral_path(
                poses, bounds, n_frames=config.render_path_frames,
                spiral_scale_f=1.0,
                spiral_scale_r=extconf['spiral_radius_scale'] if 'spiral_radius_scale' in extconf else 1.0)
        else:
            # Rotate/scale poses to align ground with xy plane and fit to unit cube.
            poses, transform, scaling_factor = camera_utils.transform_poses_pca(poses)

            transform_data = {
                'transform': transform.tolist(),  
                'scaling_factor': float(scaling_factor),
                'dataset_size': len(image_names)
            }

            transform_path = os.path.join(config.exp_path, 'transform.json')
            with open(transform_path, 'w') as f:
                json.dump(transform_data, f, indent=2)

            self.colmap_to_world_transform = transform
            if config.render_spline_keyframes is not None:
                rets = camera_utils.create_render_spline_path(config, image_names,
                                                              poses, self.exposures)
                self.spline_indices, self.render_poses, self.render_exposures = rets
            else:
                # Automatically generated inward-facing elliptical render path.
                self.render_poses = camera_utils.generate_ellipse_path(
                    poses,
                    n_frames=config.render_path_frames,
                    z_variation=config.z_variation,
                    z_phase=config.z_phase)

        # Select the split.
        all_indices = np.arange(len(image_names))
        # Remove ignored images
        ignored_imgnames_file = os.path.join(self.data_dir, 'ignored_images.txt')
        if os.path.exists(ignored_imgnames_file):
            with open(ignored_imgnames_file, 'r') as f:
                ignored_imgnames_txt = f.read()
                ignored_imgnames = ignored_imgnames_txt.split('\n')
                imgnames_idx = [imgname_idx for imgname_idx, img_name in enumerate(image_names) if img_name not in ignored_imgnames]
                all_indices = all_indices[imgnames_idx]
        if config.llff_use_all_images_for_training:
            train_indices = all_indices >= 0
        else:
            train_indices = all_indices % config.llffhold != 0
        if config.llff_use_all_images_for_testing:
            test_indices = all_indices >= 0
        else:
            test_indices = all_indices % config.llffhold == 0
        split_indices = {
            utils.DataSplit.TEST: all_indices[test_indices],
            utils.DataSplit.TRAIN: all_indices[train_indices],
        }
        indices = split_indices[self.split if not config.render_train else utils.DataSplit.TRAIN]
        image_names = [image_names[i] for i in indices]
        # Cache all poses
        all_poses = poses[all_indices]
        poses = poses[indices]
        # if self.split == utils.DataSplit.TRAIN:
        #     # load different training data on different rank
        #     local_indices = [i for i in range(len(image_names)) if (i + self.global_rank) % self.world_size == 0]
        #     image_names = [image_names[i] for i in local_indices]
        #     poses = poses[local_indices]
        #     indices = local_indices


        raw_testscene = False
        if config.rawnerf_mode:
            # Load raw images and metadata.
            images, metadata, raw_testscene = raw_utils.load_raw_dataset(
                self.split,
                self.data_dir,
                image_names,
                config.exposure_percentile,
                factor)
            self.metadata = metadata

        else:
            # Load images.
            colmap_image_dir = os.path.join(self.data_dir, 'images')
            no_factor_suffix = extconf.get('no_factor_suffix', False) or config.no_factor_suffix
            image_dir = os.path.join(self.data_dir, 'images' + image_dir_suffix) if not no_factor_suffix else colmap_image_dir
            for d in [image_dir, colmap_image_dir]:
                if not utils.file_exists(d):
                    raise ValueError(f'Image folder {d} does not exist.')
            # Downsampled images may have different names vs images used for COLMAP,
            # so we need to map between the two sorted lists of files.
            colmap_files = sorted(utils.listdir(colmap_image_dir))
            image_files = sorted(utils.listdir(image_dir))
            colmap_to_image = dict(zip(colmap_files, image_files))
            image_paths = [os.path.join(image_dir, colmap_to_image[f])
                           for f in image_names]
            images = [utils.load_img(x) for x in tqdm(image_paths, desc='Loading LLFF dataset', disable=self.global_rank != 0, leave=False)]
            images = np.stack(images, axis=0) / 255.

            # EXIF data is usually only present in the original JPEG images.
            jpeg_paths = [os.path.join(colmap_image_dir, f) for f in image_names]
            exifs = [utils.load_exif(x) for x in jpeg_paths]
            self.exifs = exifs
            if 'ExposureTime' in exifs[0] and 'ISOSpeedRatings' in exifs[0]:
                gather_exif_value = lambda k: np.array([float(x[k]) for x in exifs])
                shutters = gather_exif_value('ExposureTime')
                isos = gather_exif_value('ISOSpeedRatings')
                self.exposures = shutters * isos / 1000.
            
        if raw_testscene:
            # For raw testscene, the first image sent to COLMAP has the same pose as
            # the ground truth test image. The remaining images form the training set.
            raw_testscene_poses = {
                utils.DataSplit.TEST: poses[:1],
                utils.DataSplit.TRAIN: poses[1:],
            }
            poses = raw_testscene_poses[self.split]

        # Load images for 3D bilateral grid fine-tuning.
        if self.split == utils.DataSplit.TRAIN and config.ft_tgt_image and config.ft_tgt_pose and not config.compute_visibility:
            if isinstance(config.ft_tgt_image, str):
                ft_tgt_images = [config.ft_tgt_image]
            elif isinstance(config.ft_tgt_image, list):
                ft_tgt_images = config.ft_tgt_image
            else:
                raise ValueError(f'ft_tgt_image should be a str or list.')

            if isinstance(config.ft_tgt_pose, str):
                ft_tgt_poses = [config.ft_tgt_pose]
            elif isinstance(config.ft_tgt_pose, list):
                ft_tgt_poses = config.ft_tgt_pose
            else:
                raise ValueError(f'ft_tgt_pose should be a str or list.')

            assert len(ft_tgt_poses) == len(ft_tgt_images)
            
            images = []
            for tgt_im_path in ft_tgt_images:
                im = utils.load_img(tgt_im_path)
                im = np.array(im) / 255.
                images.append(im)
            images = np.stack(images, axis=0)
            
            poses = []
            for tgt_pose_str in ft_tgt_poses:
                # Parse target pose.
                if tgt_pose_str.find(':') >= 0:
                    tgt_pose_split, tgt_pose_idx = tgt_pose_str.split(':')
                    tgt_pose_idx = int(tgt_pose_idx)
                    if tgt_pose_split == 'all':
                        tgt_pose = all_poses[tgt_pose_idx]
                    elif tgt_pose_split == 'train':
                        tgt_pose = all_poses[np.where(train_indices == 1)[0][tgt_pose_idx]]
                    elif tgt_pose_split == 'test':
                        tgt_pose = all_poses[np.where(test_indices == 1)[0][tgt_pose_idx]]
                    elif tgt_pose_split == 'path':
                        tgt_pose = self.render_poses[tgt_pose_idx]
                    else:
                        raise ValueError(f'Cannot parse ft_tgt_pose with unrecognized split token: {tgt_pose_split}.')
                    poses.append(tgt_pose)
                else:
                    raise ValueError(f'Cannot parse ft_tgt_pose.')
            poses = np.stack(poses, axis=0)

        self.poses = poses
        self.images = images
        self.camtoworlds = self.render_poses if config.render_path else poses
        self.height, self.width = images.shape[1:3]


class TanksAndTemplesNerfPP(Dataset):
    """Subset of Tanks and Temples Dataset as processed by NeRF++."""

    def _load_renderings(self, config):
        """Load images from disk."""
        if config.render_path:
            split_str = 'camera_path'
        else:
            split_str = self.split.value

        basedir = os.path.join(self.data_dir, split_str)

        # TODO: need to rewrite this to put different data on different rank
        def load_files(dirname, load_fn, shape=None):
            files = [
                os.path.join(basedir, dirname, f)
                for f in sorted(utils.listdir(os.path.join(basedir, dirname)))
            ]
            mats = np.array([load_fn(utils.open_file(f, 'rb')) for f in files])
            if shape is not None:
                mats = mats.reshape(mats.shape[:1] + shape)
            return mats

        poses = load_files('pose', np.loadtxt, (4, 4))
        # Flip Y and Z axes to get correct coordinate frame.
        poses = np.matmul(poses, np.diag(np.array([1, -1, -1, 1])))

        # For now, ignore all but the first focal length in intrinsics
        intrinsics = load_files('intrinsics', np.loadtxt, (4, 4))

        if not config.render_path:
            images = load_files('rgb', lambda f: np.array(Image.open(f))) / 255.
            self.images = images
            self.height, self.width = self.images.shape[1:3]

        else:
            # Hack to grab the image resolution from a test image
            d = os.path.join(self.data_dir, 'test', 'rgb')
            f = os.path.join(d, sorted(utils.listdir(d))[0])
            shape = utils.load_img(f).shape
            self.height, self.width = shape[:2]
            self.images = None

        self.camtoworlds = poses
        self.focal = intrinsics[0, 0, 0]
        self.pixtocams = camera_utils.get_pixtocam(self.focal, self.width,
                                                   self.height)


class TanksAndTemplesFVS(Dataset):
    """Subset of Tanks and Temples Dataset as processed by Free View Synthesis."""

    def _load_renderings(self, config):
        """Load images from disk."""
        render_only = config.render_path and self.split == utils.DataSplit.TEST

        basedir = os.path.join(self.data_dir, 'dense')
        sizes = [f for f in sorted(utils.listdir(basedir)) if f.startswith('ibr3d')]
        sizes = sizes[::-1]

        if config.factor >= len(sizes):
            raise ValueError(f'Factor {config.factor} larger than {len(sizes)}')

        basedir = os.path.join(basedir, sizes[config.factor])
        open_fn = lambda f: utils.open_file(os.path.join(basedir, f), 'rb')

        files = [f for f in sorted(utils.listdir(basedir)) if f.startswith('im_')]
        if render_only:
            files = files[:1]
        images = np.array([np.array(Image.open(open_fn(f))) for f in files]) / 255.

        names = ['Ks', 'Rs', 'ts']
        intrinsics, rot, trans = (np.load(open_fn(f'{n}.npy')) for n in names)

        # Convert poses from colmap world-to-cam into our cam-to-world.
        w2c = np.concatenate([rot, trans[..., None]], axis=-1)
        c2w_colmap = np.linalg.inv(camera_utils.pad_poses(w2c))[:, :3, :4]
        c2w = c2w_colmap @ np.diag(np.array([1, -1, -1, 1]))

        # Reorient poses so z-axis is up
        poses, _ = camera_utils.transform_poses_pca(c2w)
        self.poses = poses

        self.images = images
        self.height, self.width = self.images.shape[1:3]
        self.camtoworlds = poses
        # For now, ignore all but the first focal length in intrinsics
        self.focal = intrinsics[0, 0, 0]
        self.pixtocams = camera_utils.get_pixtocam(self.focal, self.width,
                                                   self.height)

        if render_only:
            render_path = camera_utils.generate_ellipse_path(
                poses,
                config.render_path_frames,
                z_variation=config.z_variation,
                z_phase=config.z_phase)
            self.images = None
            self.camtoworlds = render_path
            self.render_poses = render_path
        else:
            # Select the split.
            all_indices = np.arange(images.shape[0])
            indices = {
                utils.DataSplit.TEST:
                    all_indices[all_indices % config.llffhold == 0],
                utils.DataSplit.TRAIN:
                    all_indices[all_indices % config.llffhold != 0],
            }[self.split]

            self.images = self.images[indices]
            self.camtoworlds = self.camtoworlds[indices]


class DTU(Dataset):
    """DTU Dataset."""

    def _load_renderings(self, config):
        """Load images from disk."""
        if config.render_path:
            raise ValueError('render_path cannot be used for the DTU dataset.')

        images = []
        pixtocams = []
        camtoworlds = []

        # Find out whether the particular scan has 49 or 65 images.
        n_images = len(utils.listdir(self.data_dir)) // 8

        # Loop over all images.
        for i in range(1, n_images + 1):
            # Set light condition string accordingly.
            if config.dtu_light_cond < 7:
                light_str = f'{config.dtu_light_cond}_r' + ('5000'
                                                            if i < 50 else '7000')
            else:
                light_str = 'max'

            # Load image.
            fname = os.path.join(self.data_dir, f'rect_{i:03d}_{light_str}.png')
            image = utils.load_img(fname) / 255.
            if config.factor > 1:
                image = lib_image.downsample(image, config.factor)
            images.append(image)

            # Load projection matrix from file.
            fname = os.path.join(self.data_dir, f'../../cal18/pos_{i:03d}.txt')
            with utils.open_file(fname, 'rb') as f:
                projection = np.loadtxt(f, dtype=np.float32)

            # Decompose projection matrix into pose and camera matrix.
            camera_mat, rot_mat, t = cv2.decomposeProjectionMatrix(projection)[:3]
            camera_mat = camera_mat / camera_mat[2, 2]
            pose = np.eye(4, dtype=np.float32)
            pose[:3, :3] = rot_mat.transpose()
            pose[:3, 3] = (t[:3] / t[3])[:, 0]
            pose = pose[:3]
            camtoworlds.append(pose)

            if config.factor > 0:
                # Scale camera matrix according to downsampling factor.
                camera_mat = np.diag([1. / config.factor, 1. / config.factor, 1.
                                      ]).astype(np.float32) @ camera_mat
            pixtocams.append(np.linalg.inv(camera_mat))

        pixtocams = np.stack(pixtocams)
        camtoworlds = np.stack(camtoworlds)
        images = np.stack(images)

        def rescale_poses(poses):
            """Rescales camera poses according to maximum x/y/z value."""
            s = np.max(np.abs(poses[:, :3, -1]))
            out = np.copy(poses)
            out[:, :3, -1] /= s
            return out

        # Center and scale poses.
        camtoworlds, _ = camera_utils.recenter_poses(camtoworlds)
        camtoworlds = rescale_poses(camtoworlds)
        # Flip y and z axes to get poses in OpenGL coordinate system.
        camtoworlds = camtoworlds @ np.diag([1., -1., -1., 1.]).astype(np.float32)

        all_indices = np.arange(images.shape[0])
        split_indices = {
            utils.DataSplit.TEST: all_indices[all_indices % config.dtuhold == 0],
            utils.DataSplit.TRAIN: all_indices[all_indices % config.dtuhold != 0],
        }
        indices = split_indices[self.split]

        self.images = images[indices]
        self.height, self.width = images.shape[1:3]
        self.camtoworlds = camtoworlds[indices]
        self.pixtocams = pixtocams[indices]


class Multicam(Dataset):
    def __init__(self,
                 split: str,
                 data_dir: str,
                 config: configs.Config):
        super().__init__(split, data_dir, config)

        self.multiscale_levels = config.multiscale_levels

        images, camtoworlds, pixtocams, pixtocam_ndc = \
            self.images, self.camtoworlds, self.pixtocams, self.pixtocam_ndc
        self.heights, self.widths, self.focals, self.images, self.camtoworlds, self.pixtocams, self.lossmults = [], [], [], [], [], [], []
        if pixtocam_ndc is not None:
            self.pixtocam_ndc = []
        else:
            self.pixtocam_ndc = None

        for i in range(self._n_examples):
            for j in range(self.multiscale_levels):
                self.heights.append(self.height // 2 ** j)
                self.widths.append(self.width // 2 ** j)

                self.pixtocams.append(pixtocams @ np.diag([self.height / self.heights[-1],
                                                           self.width / self.widths[-1],
                                                           1.]))
                self.focals.append(1. / self.pixtocams[-1][0, 0])
                if config.forward_facing:
                    # Set the projective matrix defining the NDC transformation.
                    self.pixtocam_ndc.append(pixtocams.reshape(3, 3))

                self.camtoworlds.append(camtoworlds[i])
                self.lossmults.append(2. ** j)
                self.images.append(self.down2(images[i], (self.heights[-1], self.widths[-1])))
        self.pixtocams = np.stack(self.pixtocams)
        self.camtoworlds = np.stack(self.camtoworlds)
        self.cameras = (self.pixtocams,
                        self.camtoworlds,
                        self.distortion_params,
                        np.stack(self.pixtocam_ndc) if self.pixtocam_ndc is not None else None)
        self._generate_rays()

        if self.split == utils.DataSplit.TRAIN:
            # Always flatten out the height x width dimensions
            def flatten(x):
                if x[0] is not None:
                    x = [y.reshape([-1, y.shape[-1]]) for y in x]
                    if self._batching == utils.BatchingMethod.ALL_IMAGES:
                        # If global batching, also concatenate all data into one list
                        x = np.concatenate(x, axis=0)
                    return x
                else:
                    return None

            self.batches = {k: flatten(v) for k, v in self.batches.items()}
        self._n_examples = len(self.camtoworlds)

        # Seed the queue with one batch to avoid race condition.
        if self.split == utils.DataSplit.TRAIN:
            self._next_fn = self._next_train
        else:
            self._next_fn = self._next_test

    def _generate_rays(self):
        if self.global_rank == 0:
            tbar = tqdm(range(len(self.camtoworlds)), desc='Generating rays', leave=False)
        else:
            tbar = range(len(self.camtoworlds))

        self.batches = defaultdict(list)
        for cam_idx in tbar:
            pix_x_int, pix_y_int = camera_utils.pixel_coordinates(
                self.widths[cam_idx], self.heights[cam_idx])
            broadcast_scalar = lambda x: np.broadcast_to(x, pix_x_int.shape)[..., None]
            ray_kwargs = {
                'lossmult': broadcast_scalar(self.lossmults[cam_idx]),
                'near': broadcast_scalar(self.near),
                'far': broadcast_scalar(self.far),
                'cam_idx': broadcast_scalar(cam_idx),
            }

            pixels = dict(pix_x_int=pix_x_int, pix_y_int=pix_y_int, **ray_kwargs)

            batch = camera_utils.cast_ray_batch(self.cameras, pixels, self.camtype)
            if not self.render_path:
                batch['rgb'] = self.images[cam_idx]
            if self._load_disps:
                batch['disps'] = self.disp_images[cam_idx, pix_y_int, pix_x_int]
            if self._load_normals:
                batch['normals'] = self.normal_images[cam_idx, pix_y_int, pix_x_int]
                batch['alphas'] = self.alphas[cam_idx, pix_y_int, pix_x_int]
            for k, v in batch.items():
                self.batches[k].append(v)

    def _next_train(self, item):
        """Sample next training batch (random rays)."""
        # We assume all images in the dataset are the same resolution, so we can use
        # the same width/height for sampling all pixels coordinates in the batch.
        # Batch/patch sampling parameters.
        num_patches = self._batch_size // self._patch_size ** 2
        # Random camera indices.
        if self._batching == utils.BatchingMethod.ALL_IMAGES:
            ray_indices = np.random.randint(0, self.batches['origins'].shape[0], (num_patches, 1, 1))
            batch = {k: v[ray_indices] if v is not None else None for k, v in self.batches.items()}
        else:
            image_index = np.random.randint(0, self._n_examples, ())
            ray_indices = np.random.randint(0, self.batches['origins'][image_index].shape[0], (num_patches,))
            batch = {k: v[image_index][ray_indices] if v is not None else None for k, v in self.batches.items()}
        batch['cam_dirs'] = -self.camtoworlds[batch['cam_idx'][..., 0]][..., 2]
        return {k: torch.from_numpy(v.copy()).float() if v is not None else None for k, v in batch.items()}

    def _next_test(self, item):
        """Sample next test batch (one full image)."""
        batch = {k: v[item] for k, v in self.batches.items()}
        batch['cam_dirs'] = -self.camtoworlds[batch['cam_idx'][..., 0]][..., 2]
        return {k: torch.from_numpy(v.copy()).float() if v is not None else None for k, v in batch.items()}

    @staticmethod
    def down2(img, sh):
        return cv2.resize(img, sh[::-1], interpolation=cv2.INTER_CUBIC)


class MultiLLFF(Multicam, LLFF):
    pass


if __name__ == '__main__':
    from internal import configs
    import accelerate

    config = configs.Config()
    accelerator = accelerate.Accelerator()
    config.world_size = accelerator.num_processes
    config.global_rank = accelerator.process_index
    config.factor = 8
    dataset = LLFF('test', '/SSD_DISK/datasets/360_v2/bicycle', config)
    print(len(dataset))
    for _ in tqdm(dataset):
        pass
    print('done')
    # print(accelerator.process_index)
