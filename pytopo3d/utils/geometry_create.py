"""
Utilities for creating masks for topology optimization.

This module provides functions to create various shapes and to parse
geometry configuration files.
"""

import json
import numpy as np
from typing import Dict, List, Tuple, Union, Optional

def create_cube(
    shape: Tuple[int, int, int],
    center: Tuple[float, float, float],
    size: Union[float, Tuple[float, float, float]]
) -> np.ndarray:
    """
    Create a cube or cuboid mask.
    
    Parameters
    ----------
    shape : tuple of int
        Shape of the design domain (nely, nelx, nelz).
    center : tuple of float
        Center coordinates of the cube as fractions [0-1] of domain size (x, y, z).
    size : float or tuple of float
        Size of the cube as fraction of domain size. If a single value, creates a cube.
        If three values, creates a cuboid with (x, y, z) dimensions.
        
    Returns
    -------
    np.ndarray
        Boolean mask with True values where the cube is located.
    """
    nely, nelx, nelz = shape
    
    # Convert center from fraction to indices
    cx = int(center[0] * nelx)
    cy = int(center[1] * nely)
    cz = int(center[2] * nelz)
    
    # Convert size from fraction to number of elements
    if isinstance(size, (int, float)):
        half_x = int(size * nelx / 2)
        half_y = int(size * nely / 2)
        half_z = int(size * nelz / 2)
    else:
        half_x = int(size[0] * nelx / 2)
        half_y = int(size[1] * nely / 2)
        half_z = int(size[2] * nelz / 2)
    
    # Calculate bounds with clipping to prevent out-of-bounds indices
    x_lo, x_hi = max(0, cx - half_x), min(nelx, cx + half_x)
    y_lo, y_hi = max(0, cy - half_y), min(nely, cy + half_y)
    z_lo, z_hi = max(0, cz - half_z), min(nelz, cz + half_z)
    
    # Create the mask with correct dimensions (nely, nelx, nelz)
    mask = np.zeros((nely, nelx, nelz), dtype=bool)
    mask[y_lo:y_hi, x_lo:x_hi, z_lo:z_hi] = True
    
    return mask

def create_sphere(
    shape: Tuple[int, int, int],
    center: Tuple[float, float, float],
    radius: float
) -> np.ndarray:
    """
    Create a spherical mask.
    
    Parameters
    ----------
    shape : tuple of int
        Shape of the design domain (nely, nelx, nelz).
    center : tuple of float
        Center coordinates of the sphere as fractions [0-1] of domain size (x, y, z).
    radius : float
        Radius of the sphere as a fraction of the smallest domain dimension.
        
    Returns
    -------
    np.ndarray
        Boolean mask with True values where the sphere is located.
    """
    nely, nelx, nelz = shape
    
    # Convert center from fraction to indices
    cx = int(center[0] * nelx)
    cy = int(center[1] * nely)
    cz = int(center[2] * nelz)
    
    # Convert radius from fraction to number of elements
    r = int(radius * min(nelx, nely, nelz))
    
    # Create coordinates relative to center
    y_indices, x_indices, z_indices = np.ogrid[:nely, :nelx, :nelz]
    y_distance = y_indices - cy
    x_distance = x_indices - cx
    z_distance = z_indices - cz
    
    # Calculate squared distance from center
    dist_squared = x_distance**2 + y_distance**2 + z_distance**2
    
    # Create the mask
    mask = np.zeros((nely, nelx, nelz), dtype=bool)
    mask[dist_squared <= r**2] = True
    
    return mask

def create_cylinder(
    shape: Tuple[int, int, int],
    center: Tuple[float, float, float],
    radius: float,
    height: float,
    axis: int = 2
) -> np.ndarray:
    """
    Create a cylindrical mask.
    
    Parameters
    ----------
    shape : tuple of int
        Shape of the design domain (nely, nelx, nelz).
    center : tuple of float
        Center coordinates of the cylinder as fractions [0-1] of domain size (x, y, z).
    radius : float
        Radius of the cylinder as a fraction of the smallest domain dimension in the
        plane perpendicular to the cylinder axis.
    height : float
        Height of the cylinder as a fraction of the domain dimension along the cylinder axis.
    axis : int, optional
        Axis along which the cylinder extends (0=x, 1=y, 2=z). Default is 2 (z-axis).
        
    Returns
    -------
    np.ndarray
        Boolean mask with True values where the cylinder is located.
    """
    nely, nelx, nelz = shape
    dimensions = [nelx, nely, nelz]
    
    # Convert center from fraction to indices
    cx = int(center[0] * nelx)
    cy = int(center[1] * nely)
    cz = int(center[2] * nelz)
    center_idx = [cx, cy, cz]
    
    # Determine perpendicular dimensions based on axis
    perp_dims = [i for i in range(3) if i != axis]
    perp_shape = [dimensions[i] for i in perp_dims]
    
    # Convert radius from fraction to number of elements
    r = int(radius * min(perp_shape))
    
    # Convert height from fraction to number of elements
    h = int(height * dimensions[axis])
    half_h = h // 2
    
    # Calculate bounds for the axis direction with clipping
    axis_lo = max(0, center_idx[axis] - half_h)
    axis_hi = min(dimensions[axis], center_idx[axis] + half_h)
    
    # Create the mask with correct dimensions
    mask = np.zeros((nely, nelx, nelz), dtype=bool)
    
    # Create coordinates relative to center
    y_indices, x_indices, z_indices = np.ogrid[:nely, :nelx, :nelz]
    y_distance = y_indices - cy
    x_distance = x_indices - cx
    z_distance = z_indices - cz
    
    if axis == 0:  # x-axis
        # Calculate distance in yz-plane
        dist_squared = y_distance**2 + z_distance**2
        # Apply mask for elements within radius and within height bounds
        x_in_bounds = (x_indices >= axis_lo) & (x_indices < axis_hi)
        mask[(dist_squared <= r**2) & x_in_bounds] = True
        
    elif axis == 1:  # y-axis
        # Calculate distance in xz-plane
        dist_squared = x_distance**2 + z_distance**2
        # Apply mask for elements within radius and within height bounds
        y_in_bounds = (y_indices >= axis_lo) & (y_indices < axis_hi)
        mask[(dist_squared <= r**2) & y_in_bounds] = True
        
    else:  # z-axis (default)
        # Calculate distance in xy-plane
        dist_squared = x_distance**2 + y_distance**2
        # Apply mask for elements within radius and within height bounds
        z_in_bounds = (z_indices >= axis_lo) & (z_indices < axis_hi)
        mask[(dist_squared <= r**2) & z_in_bounds] = True
    
    return mask

def create_geometry_from_config(
    shape: Tuple[int, int, int],
    config: Dict
) -> np.ndarray:
    """
    Create a geometry mask from a configuration dictionary.
    
    Parameters
    ----------
    shape : tuple of int
        Shape of the design domain (nely, nelx, nelz).
    config : dict
        Configuration dictionary describing the geometr.
        Must contain 'type' and other required parameters for that type.
        
    Returns
    -------
    np.ndarray
        Boolean mask with True values where the geometry is located.
    """
    geometry_type = config.get('type', '').lower()
    
    if geometry_type == 'cube':
        center = config.get('center', [0.5, 0.5, 0.5])
        size = config.get('size', 0.2)
        return create_cube(shape, center, size)
    
    elif geometry_type == 'sphere':
        center = config.get('center', [0.5, 0.5, 0.5])
        radius = config.get('radius', 0.2)
        return create_sphere(shape, center, radius)
    
    elif geometry_type == 'cylinder':
        center = config.get('center', [0.5, 0.5, 0.5])
        radius = config.get('radius', 0.2)
        height = config.get('height', 0.5)
        axis = config.get('axis', 2)
        return create_cylinder(shape, center, radius, height, axis)
    
    else:
        raise ValueError(f"Unknown geometry type: {geometry_type}")

def parse_geometry_config_file(
    config_file: str,
    shape: Tuple[int, int, int]
) -> np.ndarray:
    """
    Parse a JSON configuration file and create a geometry mask.
    
    Parameters
    ----------
    config_file : str
        Path to the JSON configuration file.
    shape : tuple of int
        Shape of the design domain (nely, nelx, nelz).
        
    Returns
    -------
    np.ndarray
        Combined boolean mask with True values where any geometry is located.
    """
    with open(config_file, 'r') as f:
        config = json.load(f)
    
    # Create a mask with all False (no geometry)
    combined_mask = np.zeros(shape, dtype=bool)
    
    # Process each geometry in the config
    geometry = config.get('geometry', [])
    for geometry_config in geometry:
        mask = create_geometry_from_config(shape, geometry_config)
        # Combine with OR operation
        combined_mask = np.logical_or(combined_mask, mask)
    
    return combined_mask 

def parse_force_config_file(
        config_file: str,
        shape: Tuple[int, int, int]
) -> np.ndarray:
    """
    Parse a JSON force configuration file and create a force field.
    
    Parameters
    ----------
    config_file : str
        Path to the JSON configuration file.
    shape : tuple of int
        Shape of the design domain (nely, nelx, nelz).
        
    Returns
    -------
    np.ndarray
        Array of xyz force values at each point in space
    """

    with open(config_file, 'r') as f:
        config = json.load(f)
    
    # Create an empty force field
    combined_field = np.zeros((shape[0], shape[1], shape[2], 3), dtype=float)
    geometry = config.get('geometry', [])

    for geometry_config in geometry:
        # Get basic binary mask for this geometry entry
        mask = create_geometry_from_config(shape, geometry_config)

        # Determine force vector for this geometry entry.
        # Supported formats in the geometry_config:
        # - 'forces': [fx, fy, fz] (list/tuple of length 3)
        if 'forces' in geometry_config:
            fval = geometry_config.get('forces')
            try:
                fvec = np.asarray(fval, dtype=float)
            except Exception:
                raise ValueError(f"Invalid force specification: {fval}")

            if fvec.size == 3:
                fvec = fvec.reshape((3,))
            else:
                raise ValueError("'force' must be a scalar or length-3 sequence")
        else:
            continue

        # Add the force vector to all positions where mask is True.
        # Use boolean indexing to broadcast the 3-component vector.
        if mask.dtype != bool:
            mask = mask.astype(bool)

        # Ensure mask shape matches the spatial dimensions
        if mask.shape != (shape[0], shape[1], shape[2]):
            raise ValueError(
                f"Geometry mask shape {mask.shape} does not match expected shape {shape}"
            )

        # Accumulate forces at masked element positions
        combined_field[mask, :] += fvec
    
    return combined_field
        



