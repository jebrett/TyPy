#!/usr/bin/env python3.9.25
# -*- coding: utf-8 -*-

"""
Module Name: main.py
Author: jbrett
Date Created: 2026-07-27
Last Modified: 2026-07-27
Description: In-progress development of a 3d phase diagram
visualization tool.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from skimage import measure

def find_line_ellipsoid_intersections(p0, v, ellipsoid):
    """
    Finds the exact intersection points of a line and a rotated ellipsoid.
    Line: P(t) = p0 + t * v
    """

    center = ellipsoid['center'].flatten()
    axes = np.diag(ellipsoid['diagonal'])
    R = ellipsoid['basis']
    # 1. Transform line base point and direction vector to ellipsoid's local frame
    p0_local = (p0 - center) @ R
    v_local = v @ R

    
    
    # 2. Set up the quadratic equation coefficients: A*t^2 + B*t + C = 0
    # Equation: (p0_local.x + t*v_local.x)^2 / a^2 + ... = 1
    inv_axes_sq = 1.0 / (axes ** 2)
    
    A = np.sum((v_local ** 2) * inv_axes_sq)
    B = 2.0 * np.sum(p0_local * v_local * inv_axes_sq)
    C = np.sum((p0_local ** 2) * inv_axes_sq) - 1.0
    
    # 3. Solve the quadratic discriminant
    discriminant = B**2 - 4*A*C
    
    if discriminant < 0:
        return [] # The line misses the ellipsoid entirely
        
    elif discriminant == 0:
        t = -B / (2*A)
        return [p0 + t * v] # Line is perfectly tangent (1 point)
        
    else:
        # Two intersection points (entry and exit)
        t1 = (-B - np.sqrt(discriminant)) / (2*A)
        t2 = (-B + np.sqrt(discriminant)) / (2*A)
        return [p0 + t1 * v, p0 + t2 * v]

def check_inside_ellipsoid(pts, ellipsoid):
    """
    Evaluates whether coordinates fall inside an arbitrary ellipsoid.
    Intended to be used for points which map to the SURFACE of another
    ellipsoid.

    Parameters:
    pts: A numpy array of shape (N, 3) containing the coordinates to check
    ellipsoid: A dictionary containing the parameters of the ellipsoid.
            Example:
                ellipsoid_info = {
                        "basis": matrixU,
                        "diagonal": matrixSigma,
                        "center": vec_center
                    }
    
    Returns:
    val: A boolean mask indicating if the point at that index is inside the ellipsoid (True) or outside (False)
    """
    # Extract ellipsoid parameters
    center = ellipsoid['center'].flatten()
    axes = np.diag(ellipsoid['diagonal'])

    # Transform coordinates back to the shape's local frame
    local_pts = (pts - center) @ ellipsoid['basis']
    val = (local_pts[:, 0] / axes[0])**2 + (local_pts[:, 1] / axes[1])**2 + (local_pts[:, 2] / axes[2])**2
    return val <= 1.05  # Slight tolerance padding prevents edge clipping

def get_ellipsoid_surface_points(ellipsoid, num_points=100):
    """
    Generates the surface coordinate cloud for a rotated ellipsoid.

    Parameters:
    ellipsoid: A dictionary containing the parameters of the ellipsoid
            Example:
                ellipsoid_info = {
                        "basis": matrixU,
                        "diagonal": matrixSigma,
                        "center": vec_center
                    }
    num_points: Number of points to generate along each parametric dimension

    Returns:
    global_pts: A numpy array of shape (N, 3) containing the surface points
    """
    u = np.linspace(0, 2 * np.pi, num_points)
    v = np.linspace(0, np.pi, num_points)
    U, V = np.meshgrid(u, v)

    # Extract ellipsoid parameters
    center = ellipsoid['center'].flatten()
    axes = np.diag(ellipsoid['diagonal'])
    
    # Standard parametric sphere mapped to the base semi-axes lengths
    X_local = axes[0] * np.sin(V) * np.cos(U)
    Y_local = axes[1] * np.sin(V) * np.sin(U)
    Z_local = axes[2] * np.cos(V)
    
    local_pts = np.vstack([X_local.ravel(), Y_local.ravel(), Z_local.ravel()]).T
    
    # Rotate and shift out into the global scene coordinates
    global_pts = (local_pts @ ellipsoid['basis'].T) + center
    return global_pts

def ellipsoid_intersect(ellipsoid1, ellipsoid2, fig, mesh_color='green'):
    '''
    Plots the intersection volume of two ellipsoids.

    Parameters:
    ellipsoid1: A dictionary containing the parameters of the first ellipsoid
        Example:
            ellipsoid_info = {
                    "basis": matrixU,
                    "diagonal": matrixSigma,
                    "center": vec_center
                }
    ellipsoid2: A dictionary containing the parameters of the second ellipsoid
    fig: A Plotly figure object to which the intersection volume will be added
    mesh_color: Color of the intersection volume mesh

    Returns:
    None
    '''
    # Extract explicit outer boundary coordinate rings for both items
    pts1 = get_ellipsoid_surface_points(ellipsoid1, num_points=120)
    pts2 = get_ellipsoid_surface_points(ellipsoid2, num_points=120)

    # Filter: Only keep points from Ellipsoid 1 that sit INSIDE Ellipsoid 2
    mask1 = check_inside_ellipsoid(pts1, ellipsoid2)
    valid_pts1 = pts1[mask1]

    # Filter: Only keep points from Ellipsoid 2 that sit INSIDE Ellipsoid 1
    mask2 = check_inside_ellipsoid(pts2, ellipsoid1)
    valid_pts2 = pts2[mask2]

    # Merge the two interlocking shell patches together
    intersection_mesh_points = np.vstack([valid_pts1, valid_pts2])

    fig.add_trace(go.Mesh3d(
        x=intersection_mesh_points[:, 0],
        y=intersection_mesh_points[:, 1],
        z=intersection_mesh_points[:, 2],
        alphahull=0,      # Automatically wraps a tight geometric skin around the points
        opacity=0.8,
        color=mesh_color,
        name='Intersection Solid'
    ))

def plot_ellipsoid(theta_rotate, phi_rotate, axes_lengths, mesh_color, fig, center=np.array([[0], [0], [0]]), rho=0, positioning='center'):
    '''
    Plots an ellipsoid in 3D space based on the provided parameters.

    Parameters:
    theta_rotate: Rotation angle in degrees around the z-axis (0 <= theta < 360)
    phi_rotate: Rotation angle in degrees from the z-axis (0 <= phi <= 180)
    axes_lengths: A list or array of the lengths of the ellipsoid's semi-axes
    mesh_color: Color of the ellipsoid mesh
    fig: A Plotly figure object to which the ellipsoid will be added
    center: Optional numpy array specifying the center of the ellipsoid (default is [0, 0, 0])
    rho: Optional parameter for positioning the ellipsoid (default is 0)
    positioning: Optional parameter for specifying the positioning method (default is 'center')

    Returns:
    ellipsoid_info: A dictionary containing the orthonormal basis, diagonal matrix, and center of the ellipsoid
    '''
    # 1. Define the orthonormal basis for the principal axes of the ellipsoid

    # Begin with theta (rotation in x-y plane from x-axis, 0<=theta<=360)
    # and phi (angle from z-axis, 0<=phi<=180)
    theta_angle = np.radians(0 + theta_rotate)
    phi_angle = np.radians(90 - phi_rotate)
    # Define vector vec1
    vec1 = np.array([np.cos(theta_angle) * np.sin(phi_angle),
                    np.sin(theta_angle) * np.sin(phi_angle),
                    np.cos(phi_angle)])

    # Define vector vec2 and normalize it
    vec2 = np.linalg.cross(np.array([0, 0, 1]), vec1)
    vec2 /= np.linalg.norm(vec2)

    # Define vector vec3 as the cross product of vec1 and vec2
    vec3 = np.linalg.cross(vec1, vec2)

    # Use vec1, 2, and 3 to define orthonormal basis
    matrixU = np.column_stack((vec1, vec2, vec3))

    # 2. Define center of ellipsoid as explicit column vector
    if positioning == 'center':
        vec_center = center[:, np.newaxis]
    elif positioning == 'vector':
        vec_center = rho * vec1[:, np.newaxis]
    else:
        raise ValueError("Invalid positioning method. Use 'center' or 'vector'.")
        return 0

    # 3. Define principal axes lengths of the ellipsoid in Sigma matrix,
    # then calculate matrix A to define ellipsoid
    matrixSigma = np.diag(axes_lengths)
    matrixA = matrixU @ (matrixSigma**2) @ matrixU.T

    # 4. Calculate eigenvalues and eigenvectors
    eigenvalues, eigenvectors = np.linalg.eigh(matrixA)

    # Radii of the ellipsoid (semi-axes lengths)
    radii = np.sqrt(eigenvalues)

    # 5. Generate coordinates for a standard unit sphere, scaled for axes lengths
    u = np.linspace(0, 2 * np.pi, 100)
    v = np.linspace(0, np.pi, 100)

    x_scaled = radii[0] * np.outer(np.cos(u), np.sin(v))
    y_scaled = radii[1] * np.outer(np.sin(u), np.sin(v))
    z_scaled = radii[2] * np.outer(np.ones_like(u), np.cos(v))

    # 6. Rotate using the eigenvectors matrix mapping
    x_ell = eigenvectors[0, 0]*x_scaled + eigenvectors[0, 1]*y_scaled + eigenvectors[0, 2]*z_scaled + vec_center[0]
    y_ell = eigenvectors[1, 0]*x_scaled + eigenvectors[1, 1]*y_scaled + eigenvectors[1, 2]*z_scaled + vec_center[1]
    z_ell = eigenvectors[2, 0]*x_scaled + eigenvectors[2, 1]*y_scaled + eigenvectors[2, 2]*z_scaled + vec_center[2]

    # Trace 1: Add the ellipsoid surface mesh
    fig.add_trace(go.Surface(
        x=x_ell, y=y_ell, z=z_ell,
        colorscale=[[0, mesh_color], [1, mesh_color]],
        opacity=0.3,
        showscale=False,
        contours=dict(
        x=dict(show=True, color='black', width=16),
        y=dict(show=True, color='black', width=16),
        z=dict(show=True, color='black', width=16)
        )
    ))
    
    # Trace 2: The center point of the ellipsoid
    '''fig.add_trace(go.Scatter3d(
        x=vec_center[0], 
        y=vec_center[1], 
        z=vec_center[2],
        mode='markers+text',
        marker=dict(size=4, color=mesh_color, symbol='circle'),
        name='Center',
        text=["Center"],
        textposition="top center"
    ))'''

    # Trace 3: Line from origin to ellipsoid center
    #fig.add_trace(go.Scatter3d(
    #    x=[0, vec_center[0][0]],
    #    y=[0, vec_center[1][0]],
    #    z=[0, vec_center[2][0]],
    #    mode='lines+markers', # Adds a small dot at both the origin and destination ends
    #    line=dict(color=mesh_color, width=5),
    #    marker=dict(size=4, color=mesh_color),
    #    name='Origin to Center'
    #))

    ellipsoid_info = {
        "basis": matrixU,
        "diagonal": matrixSigma,
        "center": vec_center
    }

    return ellipsoid_info

# Create Interactive 3D Surface Figure using Graph Objects
fig = go.Figure()

fig.add_trace(go.Scatter3d(
        x=[0], 
        y=[0], 
        z=[0],
        mode='markers+text',
        marker=dict(size=1, color='black', symbol='circle'),
        name='Origin',
        text=["Origin"],
        textposition="top center"
    ))

associative_reg = plot_ellipsoid(45, 15, [3, 1.5, 1.5], 'red', fig, rho=5, positioning='vector')
segregative_reg = plot_ellipsoid(45, 15, [0.5, 1.75, 1.75], 'blue', fig, rho=7.75, positioning='vector')

ellipsoid_intersect(associative_reg, segregative_reg, fig, mesh_color='green')

point1 = np.array([[1.5], [1.5], [0.6]])

fig.add_trace(go.Scatter3d(
        x=point1[0], 
        y=point1[1], 
        z=point1[2],
        mode='markers+text',
        marker=dict(size=1, color='black', symbol='circle'),
        name='Input',
        text=["Input"],
        textposition="top center"
    ))

if check_inside_ellipsoid(point1.T, associative_reg):
    
    tie_vec = point1/np.linalg.norm(point1)
    intersection_points = find_line_ellipsoid_intersections(point1.flatten(), tie_vec.flatten(), associative_reg)
    
    fig.add_trace(go.Scatter3d(
        x=[intersection_points[0][0],
        intersection_points[1][0]],
        y=[intersection_points[0][1],
        intersection_points[1][1]],
        z=[intersection_points[0][2],
        intersection_points[1][2]],
        mode='lines',
        line=dict(color='cyan', width=8),
        name=f'associative intersection'
    ))

point2 = np.array([[intersection_points[1][0]], [intersection_points[1][1]], [intersection_points[1][2]]])
print(point2)
if check_inside_ellipsoid(point2.T, segregative_reg):
    print('yup)')
    tie_vec = np.array([0.3, 0.3, -1])
    tie_vec = tie_vec/np.linalg.norm(tie_vec)
    intersection_points2 = find_line_ellipsoid_intersections(point2.flatten(), tie_vec.flatten(), segregative_reg)
    fig.add_trace(go.Scatter3d(
            x=[intersection_points2[0][0],
            intersection_points2[1][0]],
            y=[intersection_points2[0][1],
            intersection_points2[1][1]],
            z=[intersection_points2[0][2],
            intersection_points2[1][2]],
            mode='lines',
            line=dict(color='magenta', width=8),
            name=f'segregative intersection'
        ))
    
# Configure scene options and maintain forced 1:1:1 aspect ratio scaling
fig.update_layout(
    title=f"Interactive Ellipsoid Matrix Transformation",
    scene=dict(
        xaxis_title='Unmodified Chromatin',
        yaxis_title='H3K9me3 Chromatin',
        zaxis_title='Hp1alpha',
        aspectmode='data' # Keeps the 3D aspect ratio proportional
    ),
    margin=dict(l=0, r=0, b=0, t=40)
)

# Render plot in browser window
fig.show()