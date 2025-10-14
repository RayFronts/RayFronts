"""Utils to perform conversions between ros messages and arrays."""
# Code below is mostly from (https://github.com/Box-Robotics/ros2_numpy)
# Added as a single file for simplicity and because the package is not available
# on PyPI. Submodule would have been too much for such a simple package.

# Software License Agreement (BSD License)
#
# Copyright (c) 2008, Willow Garage, Inc.
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above
#    copyright notice, this list of conditions and the following
#    disclaimer in the documentation and/or other materials provided
#    with the distribution.
#  * Neither the name of Willow Garage, Inc. nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
# FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
# COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
# INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
# BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
# LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
# ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.
#
# Author: Jon Binney

import sys

import numpy as np
import array
from scipy.spatial.transform import Rotation
import cv2

from sensor_msgs.msg import PointCloud2, PointField, Image
from geometry_msgs.msg import Vector3, Quaternion, Point, Pose

## image.py

name_to_dtypes = {
    "rgb8":    (np.uint8,  3),
    "rgba8":   (np.uint8,  4),
    "rgb16":   (np.uint16, 3),
    "rgba16":  (np.uint16, 4),
    "bgr8":    (np.uint8,  3),
    "bgra8":   (np.uint8,  4),
    "bgr16":   (np.uint16, 3),
    "bgra16":  (np.uint16, 4),
    "mono8":   (np.uint8,  1),
    "mono16":  (np.uint16, 1),

    # for bayer image (based on cv_bridge.cpp)
    "bayer_rggb8":  (np.uint8,  1),
    "bayer_bggr8":  (np.uint8,  1),
    "bayer_gbrg8":  (np.uint8,  1),
    "bayer_grbg8":  (np.uint8,  1),
    "bayer_rggb16":     (np.uint16, 1),
    "bayer_bggr16":     (np.uint16, 1),
    "bayer_gbrg16":     (np.uint16, 1),
    "bayer_grbg16":     (np.uint16, 1),

    # OpenCV CvMat types
    "8UC1":    (np.uint8,   1),
    "8UC2":    (np.uint8,   2),
    "8UC3":    (np.uint8,   3),
    "8UC4":    (np.uint8,   4),
    "8SC1":    (np.int8,    1),
    "8SC2":    (np.int8,    2),
    "8SC3":    (np.int8,    3),
    "8SC4":    (np.int8,    4),
    "16UC1":   (np.uint16,   1),
    "16UC2":   (np.uint16,   2),
    "16UC3":   (np.uint16,   3),
    "16UC4":   (np.uint16,   4),
    "16SC1":   (np.int16,  1),
    "16SC2":   (np.int16,  2),
    "16SC3":   (np.int16,  3),
    "16SC4":   (np.int16,  4),
    "32SC1":   (np.int32,   1),
    "32SC2":   (np.int32,   2),
    "32SC3":   (np.int32,   3),
    "32SC4":   (np.int32,   4),
    "32FC1":   (np.float32, 1),
    "32FC2":   (np.float32, 2),
    "32FC3":   (np.float32, 3),
    "32FC4":   (np.float32, 4),
    "64FC1":   (np.float64, 1),
    "64FC2":   (np.float64, 2),
    "64FC3":   (np.float64, 3),
    "64FC4":   (np.float64, 4)
}

def image_to_numpy(msg):
    if not msg.encoding in name_to_dtypes:
        raise TypeError('Unrecognized encoding {}'.format(msg.encoding))

    dtype_class, channels = name_to_dtypes[msg.encoding]
    dtype = np.dtype(dtype_class)
    dtype = dtype.newbyteorder('>' if msg.is_bigendian else '<')
    shape = (msg.height, msg.width, channels)

    data = np.frombuffer(msg.data, dtype=dtype).reshape(shape)
    data.strides = (
        msg.step,
        dtype.itemsize * channels,
        dtype.itemsize
    )

    if channels == 1:
        data = data[...,0]
    return data

def compressed_image_to_numpy(msg_compressed):
    data = np.frombuffer(msg_compressed.data, np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_UNCHANGED)

def numpy_to_image(arr, encoding) -> Image:
    if not encoding in name_to_dtypes:
        raise TypeError('Unrecognized encoding {}'.format(encoding))

    im = Image(encoding=encoding)

    # extract width, height, and channels
    dtype_class, exp_channels = name_to_dtypes[encoding]
    dtype = np.dtype(dtype_class)
    if len(arr.shape) == 2:
        im.height, im.width, channels = arr.shape + (1,)
    elif len(arr.shape) == 3:
        im.height, im.width, channels = arr.shape
    else:
        raise TypeError("Array must be two or three dimensional")

    # check type and channels
    if exp_channels != channels:
        raise TypeError("Array has {} channels, {} requires {}".format(
            channels, encoding, exp_channels
        ))
    if dtype_class != arr.dtype.type:
        raise TypeError("Array is {}, {} requires {}".format(
            arr.dtype.type, encoding, dtype_class
        ))

    # make the array contiguous in memory, as mostly required by the format
    contig = np.ascontiguousarray(arr)
    im.data = contig.tostring()
    im.step = contig.strides[0]
    im.is_bigendian = (
        arr.dtype.byteorder == '>' or
        arr.dtype.byteorder == '=' and sys.byteorder == 'big'
    )

    return im

## point_cloud2.py

# prefix to the names of dummy fields we add to get byte alignment
# correct. this needs to not clash with any actual field names
DUMMY_FIELD_PREFIX = '__'

# mappings between PointField types and numpy types
type_mappings = [(PointField.INT8, np.dtype('int8')),
                 (PointField.UINT8, np.dtype('uint8')),
                 (PointField.INT16, np.dtype('int16')),
                 (PointField.UINT16, np.dtype('uint16')),
                 (PointField.INT32, np.dtype('int32')),
                 (PointField.UINT32, np.dtype('uint32')),
                 (PointField.FLOAT32, np.dtype('float32')),
                 (PointField.FLOAT64, np.dtype('float64'))]
pftype_to_nptype = dict(type_mappings)
nptype_to_pftype = dict((nptype, pftype) for pftype, nptype in type_mappings)

def fields_to_dtype(fields, point_step):
    '''Convert a list of PointFields to a numpy record datatype.
    '''
    offset = 0
    np_dtype_list = []
    for f in fields:
        while offset < f.offset:
            # might be extra padding between fields
            np_dtype_list.append(
                ('%s%d' % (DUMMY_FIELD_PREFIX, offset), np.uint8))
            offset += 1

        dtype = pftype_to_nptype[f.datatype]
        if f.count != 1:
            dtype = np.dtype((dtype, f.count))

        np_dtype_list.append((f.name, dtype))
        offset += pftype_to_nptype[f.datatype].itemsize * f.count

    # might be extra padding between points
    while offset < point_step:
        np_dtype_list.append(('%s%d' % (DUMMY_FIELD_PREFIX, offset), np.uint8))
        offset += 1

    return np_dtype_list

def dtype_to_fields(dtype):
    '''Convert a numpy record datatype into a list of PointFields.
    '''
    fields = []
    for field_name in dtype.names:
        np_field_type, field_offset = dtype.fields[field_name]
        pf = PointField()
        pf.name = field_name
        if np_field_type.subdtype:
            item_dtype, shape = np_field_type.subdtype
            pf.count = int(np.prod(shape))
            np_field_type = item_dtype
        else:
            pf.count = 1

        pf.datatype = nptype_to_pftype[np_field_type]
        pf.offset = field_offset
        fields.append(pf)
    return fields

def pointcloud2_to_array(cloud_msg, squeeze=True):
    ''' Converts a rospy PointCloud2 message to a numpy recordarray

    Reshapes the returned array to have shape (height, width), even if the
    height is 1.

    The reason for using np.frombuffer rather than struct.unpack is
    speed... especially for large point clouds, this will be <much> faster.
    '''
    # construct a numpy record type equivalent to the point type of this cloud
    dtype_list = fields_to_dtype(cloud_msg.fields, cloud_msg.point_step)

    # parse the cloud into an array
    cloud_arr = np.frombuffer(cloud_msg.data, dtype_list)

    # remove the dummy fields that were added
    cloud_arr = cloud_arr[
        [fname for fname, _type in dtype_list if not (
            fname[:len(DUMMY_FIELD_PREFIX)] == DUMMY_FIELD_PREFIX)]]

    if squeeze and cloud_msg.height == 1:
        return np.reshape(cloud_arr, (cloud_msg.width,))
    else:
        return np.reshape(cloud_arr, (cloud_msg.height, cloud_msg.width))

def array_to_pointcloud2(cloud_arr, stamp=None, frame_id=None):
    '''Converts a numpy record array to a sensor_msgs.msg.PointCloud2.
    '''
    # make it 2d (even if height will be 1)
    cloud_arr = np.atleast_2d(cloud_arr)

    cloud_msg = PointCloud2()

    if stamp is not None:
        cloud_msg.header.stamp = stamp
    if frame_id is not None:
        cloud_msg.header.frame_id = frame_id
    cloud_msg.height = cloud_arr.shape[0]
    cloud_msg.width = cloud_arr.shape[1]
    cloud_msg.fields = dtype_to_fields(cloud_arr.dtype)
    cloud_msg.is_bigendian = sys.byteorder != 'little'
    cloud_msg.point_step = cloud_arr.dtype.itemsize
    cloud_msg.row_step = cloud_msg.point_step*cloud_arr.shape[1]
    cloud_msg.is_dense = \
      all([np.isfinite(
            cloud_arr[fname]).all() for fname in cloud_arr.dtype.names])

    # The PointCloud2.data setter will create an array.array object for you if you don't
    # provide it one directly. This causes very slow performance because it iterates
    # over each byte in python.
    # Here we create an array.array object using a memoryview, limiting copying and
    # increasing performance.
    memory_view = memoryview(cloud_arr)
    if memory_view.nbytes > 0:
        array_bytes = memory_view.cast("B")
    else:
        # Casting raises a TypeError if the array has no elements
        array_bytes = b""
    as_array = array.array("B")
    as_array.frombytes(array_bytes)
    cloud_msg.data = as_array
    return cloud_msg

def merge_rgb_fields(cloud_arr):
    '''Takes an array with named np.uint8 fields 'r', 'g', and 'b', and returns
       an array in which they have been merged into a single np.float32 'rgb'
       field. The first byte of this field is the 'r' uint8, the second is the
       'g', uint8, and the third is the 'b' uint8.

       This is the way that pcl likes to handle RGB colors for some reason.
    '''
    r = np.asarray(cloud_arr['r'], dtype=np.uint32)
    g = np.asarray(cloud_arr['g'], dtype=np.uint32)
    b = np.asarray(cloud_arr['b'], dtype=np.uint32)
    rgb_arr = np.array((r << 16) | (g << 8) | (b << 0), dtype=np.uint32)

    # not sure if there is a better way to do this. i'm changing the type of
    # the array from uint32 to float32, but i don't want any conversion to take
    # place -jdb
    rgb_arr.dtype = np.float32

    # create a new array, without r, g, and b, but with rgb float32 field
    new_dtype = []
    for field_name in cloud_arr.dtype.names:
        field_type, field_offset = cloud_arr.dtype.fields[field_name]
        if field_name not in ('r', 'g', 'b'):
            new_dtype.append((field_name, field_type))
    new_dtype.append(('rgb', np.float32))
    new_cloud_arr = np.zeros(cloud_arr.shape, new_dtype)

    # fill in the new array
    for field_name in new_cloud_arr.dtype.names:
        if field_name == 'rgb':
            new_cloud_arr[field_name] = rgb_arr
        else:
            new_cloud_arr[field_name] = cloud_arr[field_name]

    return new_cloud_arr

def split_rgb_field(cloud_arr):
    '''Takes an array with a named 'rgb' float32 field, and returns an array in
    which this has been split into 3 uint 8 fields: 'r', 'g', and 'b'.

    (pcl stores rgb in packed 32 bit floats)
    '''
    rgb_arr = cloud_arr['rgb'].copy()
    rgb_arr.dtype = np.uint32
    r = np.asarray((rgb_arr >> 16) & 255, dtype=np.uint8)
    g = np.asarray((rgb_arr >> 8) & 255, dtype=np.uint8)
    b = np.asarray(rgb_arr & 255, dtype=np.uint8)

    # create a new array, without rgb, but with r, g, and b fields
    new_dtype = []
    for field_name in cloud_arr.dtype.names:
        field_type, field_offset = cloud_arr.dtype.fields[field_name]
        if not field_name == 'rgb':
            new_dtype.append((field_name, field_type))
    new_dtype.append(('r', np.uint8))
    new_dtype.append(('g', np.uint8))
    new_dtype.append(('b', np.uint8))
    new_cloud_arr = np.zeros(cloud_arr.shape, new_dtype)

    # fill in the new array
    for field_name in new_cloud_arr.dtype.names:
        if field_name == 'r':
            new_cloud_arr[field_name] = r
        elif field_name == 'g':
            new_cloud_arr[field_name] = g
        elif field_name == 'b':
            new_cloud_arr[field_name] = b
        else:
            new_cloud_arr[field_name] = cloud_arr[field_name]
    return new_cloud_arr

def get_xyz_points(cloud_array, remove_nans=True, dtype=float):
  '''Pulls out x, y, and z columns from the cloud recordarray, and returns
  a 3xN matrix.
  '''
  # remove crap points
  if remove_nans:
    mask = np.isfinite(cloud_array['x']) & \
           np.isfinite(cloud_array['y']) & \
           np.isfinite(cloud_array['z'])
    cloud_array = cloud_array[mask]

  # pull out x, y, and z values
  points = np.zeros(cloud_array.shape + (3,), dtype=dtype)
  points[...,0] = cloud_array['x']
  points[...,1] = cloud_array['y']
  points[...,2] = cloud_array['z']

  return points

def pointcloud2_to_xyz_array(cloud_msg, remove_nans=True):
  return get_xyz_points(
    pointcloud2_to_array(cloud_msg), remove_nans=remove_nans)

## geoemetry.py

# basic types

def vector3_to_numpy(msg, hom=False):
  if hom:
    return np.array([msg.x, msg.y, msg.z, 0])
  else:
    return np.array([msg.x, msg.y, msg.z])

def numpy_to_vector3(arr):
  if arr.shape[-1] == 4:
    assert np.all(arr[...,-1] == 0)
    arr = arr[...,:-1]
  arr = arr.astype("float")
  if len(arr.shape) == 1:
    return Vector3(**dict(zip(['x', 'y', 'z'], arr)))
  else:
    return np.apply_along_axis(
      lambda v: Vector3(**dict(zip(['x', 'y', 'z'], v))), axis=-1,
      arr=arr)

def quat_to_numpy(msg):
  return np.array([msg.x, msg.y, msg.z, msg.w])

def numpy_to_quat(arr):
  assert arr.shape[-1] == 4

  if len(arr.shape) == 1:
    return Quaternion(**dict(zip(['x', 'y', 'z', 'w'], arr)))
  else:
    return np.apply_along_axis(
      lambda v: Quaternion(**dict(zip(['x', 'y', 'z', 'w'], v))),
      axis=-1, arr=arr)

def pose_to_numpy(ros_pose):
  pose_t = vector3_to_numpy(ros_pose.position)
  pose_q = quat_to_numpy(ros_pose.orientation)
  pose_R = Rotation.from_quat(pose_q, scalar_first=False).as_matrix()
  pose_Rt_3x4 = np.concatenate((pose_R, pose_t.reshape(3, 1)), axis=1)
  row = np.array([[0,0,0,1]])
  pose_Rt_4x4 = np.concatenate((pose_Rt_3x4, row), axis=-2)
  return pose_Rt_4x4

def numpy_to_pose(pose_4x4):
  pose_R = pose_4x4[:3,:3].astype("float")
  pose_t = pose_4x4[:3, 3].astype("float")
  pose_q = Rotation.from_matrix(pose_R).as_quat(scalar_first=False)
  pose = Pose()
  pose.position = Point(x=pose_t[0], y=pose_t[1], z=pose_t[2])
  pose.orientation = numpy_to_quat(pose_q)
  return pose
