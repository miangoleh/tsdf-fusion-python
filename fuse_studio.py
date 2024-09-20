"""Fuse 1000 RGB-D images from the 7-scenes dataset into a TSDF voxel volume with 2cm resolution.
"""

import time

import cv2
import numpy as np
import os
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"


import fusion


if __name__ == "__main__":
  # ======================================================================================================== #
  # (Optional) This is an example of how to compute the 3D bounds
  # in world coordinates of the convex hull of all camera view
  # frustums in the dataset
  # ======================================================================================================== #
  print("Estimating voxel volume bounds...")
  n_imgs = 12

  # datapath = "/studio-storage1/datasets/objaverse_CPlabrender/objaverse-xl/scripts/rendering/results/renders_sketchfab_1000-8000/ab01bb17-be3b-5ad2-9795-f6fb8c327016/"
  datapath = "/studio-storage1/datasets/objaverse_CPlabrender/objaverse-xl/scripts/rendering/results/renders_sketchfab_1-1000/023340eb-a487-5270-9c82-57759fc6e45b"
  depth_gt = False
  depth_est_path = "/project/aksoy-lab/Mahdi/MultiViewFromMono/Mytestdata/Objaverse_Scratch1/renders_sketchfab_1-1000_023340eb-a487-5270-9c82-57759fc6e45b"

  # (3*3) numpy array
  vol_bnds = np.zeros((3,2))
  for i in range(n_imgs):
    camera_params = np.load(os.path.join(datapath, f"{i:03d}.npz"))
    cam_intr = camera_params['cv_matrix']
    extrinsic_matrix = camera_params['rt_matrix']
    R = extrinsic_matrix[:3, :3]
    T = extrinsic_matrix[:3, 3]
    # Invert the extrinsic matrix to get the pose matrix
    R_inv = R.T  # Transpose of rotation matrix
    T_inv = -np.dot(R_inv, T)  # Transformed translation
    # Construct the 4x4 pose matrix
    cam_pose = np.eye(4)
    cam_pose[:3, :3] = R_inv
    cam_pose[:3, 3] = T_inv
    # Read depth image and camera pose
    if depth_gt == True:
      depth_im = cv2.imread(os.path.join(datapath,f"{i:03d}_depth0001.exr"),-1).astype(float)
      depth_im[depth_im > 10] = 0  # set invalid depth to 0 (specific to 7-scenes dataset)
      depth_im = depth_im[:,:,0]
    else:
      depth_im = np.load(f"{depth_est_path}_{i:03d}_depth.npy")
    

    # (4*4) rigid transformation matrix
    # Compute camera view frustum and extend convex hull
    view_frust_pts = fusion.get_view_frustum(depth_im, cam_intr, cam_pose)
    vol_bnds[:,0] = np.minimum(vol_bnds[:,0], np.amin(view_frust_pts, axis=1))
    vol_bnds[:,1] = np.maximum(vol_bnds[:,1], np.amax(view_frust_pts, axis=1))
  # ======================================================================================================== #

  # ======================================================================================================== #
  # Integrate
  # ======================================================================================================== #
  # Initialize voxel volume
  print("Initializing voxel volume...")
  tsdf_vol = fusion.TSDFVolume(vol_bnds, voxel_size=0.005)

  # Loop through RGB-D images and fuse them together
  t0_elapse = time.time()
  for i in range(n_imgs):
    print("Fusing frame %d/%d"%(i+1, n_imgs))

    # Read RGB-D image and camera pose
    color_image = cv2.cvtColor(cv2.imread(os.path.join(datapath,f"{i:03d}.png")), cv2.COLOR_BGR2RGB)
    if depth_gt == True:
      depth_im = cv2.imread(os.path.join(datapath,f"{i:03d}_depth0001.exr"),-1).astype(float)
      depth_im[depth_im > 10] = 0
      depth_im = depth_im[:,:,0]
    else:
      depth_im = np.load(f"{depth_est_path}_{i:03d}_depth.npy")

    camera_params = np.load(os.path.join(datapath, f"{i:03d}.npz"))
    cam_intr = camera_params['cv_matrix']
    extrinsic_matrix = camera_params['rt_matrix']
    
    R = extrinsic_matrix[:3, :3]
    T = extrinsic_matrix[:3, 3]

    # Invert the extrinsic matrix to get the pose matrix
    R_inv = R.T  # Transpose of rotation matrix
    T_inv = -np.dot(R_inv, T)  # Transformed translation

    # Construct the 4x4 pose matrix
    cam_pose = np.eye(4)
    cam_pose[:3, :3] = R_inv
    cam_pose[:3, 3] = T_inv
    # Integrate observation into voxel volume (assume color aligned with depth)
    tsdf_vol.integrate(color_image, depth_im, cam_intr, cam_pose, obs_weight=1.)

  fps = n_imgs / (time.time() - t0_elapse)
  print("Average FPS: {:.2f}".format(fps))

  # Get mesh from voxel volume and save to disk (can be viewed with Meshlab)
  print("Saving mesh to mesh.ply...")
  verts, faces, norms, colors = tsdf_vol.get_mesh()
  fusion.meshwrite("mesh.ply", verts, faces, norms, colors)

  # Get point cloud from voxel volume and save to disk (can be viewed with Meshlab)
  print("Saving point cloud to pc.ply...")
  point_cloud = tsdf_vol.get_point_cloud()
  fusion.pcwrite("pc.ply", point_cloud)