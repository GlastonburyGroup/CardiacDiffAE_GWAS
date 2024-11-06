import h5py
import zarr
import os

def hdf5_to_zarr(hdf5_path, zarr_path):
    """Convert an HDF5 file to Zarr format."""

    os.makedirs(os.path.dirname(zarr_path), exist_ok=True)

    def copy(name, node):
        """Copy a node from the HDF5 file to the Zarr file."""
        if isinstance(node, h5py.Dataset):
            zarr_group[name] = node[...]
            zarr_group[name].attrs.update(node.attrs)
        else:
            zarr_group.create_group(name)
            zarr_group[name].attrs.update(node.attrs)

    with h5py.File(hdf5_path, 'r') as hdf5_file:
        zarr_group = zarr.open_group(zarr_path, mode='w')
        hdf5_file.visititems(copy)


# Usage:
hdf5_to_zarr('../ukbbH5s/F20208_Long_axis_heart_images_DICOM_H5v1/data.h5', '../ukbbZarrs/F20208_Long_axis_heart_images_DICOM_H5v1/data.zarr')
