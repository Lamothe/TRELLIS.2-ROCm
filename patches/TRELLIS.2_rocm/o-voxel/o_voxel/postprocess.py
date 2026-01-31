from typing import *
from tqdm import tqdm
import numpy as np
import torch
import trimesh
import cumesh
import scipy.spatial

def to_glb(
    vertices: torch.Tensor,
    faces: torch.Tensor,
    attr_volume: torch.Tensor,
    coords: torch.Tensor,
    attr_layout: Dict[str, slice],
    aabb: Union[list, tuple, np.ndarray, torch.Tensor],
    voxel_size: Union[float, list, tuple, np.ndarray, torch.Tensor] = None,
    grid_size: Union[int, list, tuple, np.ndarray, torch.Tensor] = None,
    decimation_target: int = 1000000,
    texture_size: int = 2048,
    remesh: bool = False,
    remesh_band: float = 1,
    remesh_project: float = 0.9,
    mesh_cluster_threshold_cone_half_angle_rad=np.radians(90.0),
    mesh_cluster_refine_iterations=0,
    mesh_cluster_global_iterations=1,
    mesh_cluster_smooth_strength=1,
    verbose: bool = False,
    use_tqdm: bool = False,
):
    print("DEBUG: Starting 'Vertex Color' Export Mode (Strix Halo Safe)")

    # 1. Input Sanitization
    vertices = torch.nan_to_num(vertices, nan=0.0).float()
    
    if isinstance(aabb, (list, tuple)): aabb = np.array(aabb)
    if isinstance(aabb, np.ndarray): aabb = torch.tensor(aabb, dtype=torch.float32, device=coords.device)
    
    if voxel_size is not None:
        if isinstance(voxel_size, float): voxel_size = [voxel_size]*3
        if isinstance(voxel_size, (list, tuple)): voxel_size = np.array(voxel_size)
        if isinstance(voxel_size, np.ndarray): voxel_size = torch.tensor(voxel_size, dtype=torch.float32, device=coords.device)
        grid_size = ((aabb[1] - aabb[0]) / voxel_size).round().int()
    else:
        if isinstance(grid_size, int): grid_size = [grid_size]*3
        if isinstance(grid_size, (list, tuple)): grid_size = np.array(grid_size)
        if isinstance(grid_size, np.ndarray): grid_size = torch.tensor(grid_size, dtype=torch.int32, device=coords.device)
        voxel_size = (aabb[1] - aabb[0]) / grid_size
    
    if use_tqdm: pbar = tqdm(total=5, desc="Processing")

    # 2. Geometry
    vertices = vertices.cuda()
    faces = faces.cuda()
    mesh = cumesh.CuMesh()
    mesh.init(vertices, faces)
    mesh.fill_holes(max_hole_perimeter=3e-2)
    _ = mesh.read()
    if use_tqdm: pbar.update(1)
        
    if not remesh:
        mesh.simplify(decimation_target * 3, verbose=verbose)
        mesh.remove_duplicate_faces()
        mesh.repair_non_manifold_edges()
        mesh.remove_small_connected_components(1e-5)
        mesh.fill_holes(max_hole_perimeter=3e-2)
        mesh.simplify(decimation_target, verbose=verbose)
        mesh.remove_duplicate_faces()
        mesh.repair_non_manifold_edges()
        mesh.remove_small_connected_components(1e-5)
        mesh.fill_holes(max_hole_perimeter=3e-2)
        mesh.unify_face_orientations()
    else:
        bvh = cumesh.cuBVH(vertices, faces)
        center = aabb.mean(dim=0)
        scale = (aabb[1] - aabb[0]).max().item()
        resolution = grid_size.max().item()
        mesh.init(*cumesh.remeshing.remesh_narrow_band_dc(
            vertices, faces, center=center, scale=(resolution + 3 * remesh_band) / resolution * scale,
            resolution=resolution, band=remesh_band, project_back=remesh_project, verbose=verbose, bvh=bvh,
        ))
        mesh.simplify(decimation_target, verbose=verbose)
    if use_tqdm: pbar.update(1)

    out_vertices_gpu, out_faces_gpu = mesh.read()
    
    # 3. VERTEX COLOR SAMPLING (CPU KDTree)
    src_pts = coords.cpu().float().numpy()
    src_feats = torch.nan_to_num(attr_volume, nan=0.0).cpu().float().numpy()
    
    mesh_verts_cpu = out_vertices_gpu.cpu().float()
    query_pts = ((mesh_verts_cpu - aabb.cpu()[0]) / voxel_size.cpu()).numpy()
    query_pts = np.nan_to_num(query_pts, nan=0.0)
    
    tree = scipy.spatial.cKDTree(src_pts, balanced_tree=False)
    dists, idxs = tree.query(query_pts, k=1)
    
    vertex_colors = src_feats[idxs]
    if use_tqdm: pbar.update(1)

    # 4. Color Processing
    try:
        color_slice = attr_layout['base_color']
        raw_rgb = vertex_colors[:, color_slice]
    except:
        raw_rgb = vertex_colors[:, :3]
        
    d_max = raw_rgb.max()
    if d_max > 0:
        if d_max < 1.0: 
             final_colors = np.clip(raw_rgb * 255, 0, 255).astype(np.uint8)
        else: 
             final_colors = np.clip(raw_rgb, 0, 255).astype(np.uint8)
    else:
        final_colors = np.full_like(raw_rgb, 200, dtype=np.uint8)

    alpha = np.full((len(final_colors), 1), 255, dtype=np.uint8)
    vertex_colors_rgba = np.concatenate([final_colors, alpha], axis=1)

    if use_tqdm: pbar.update(1)

    # 5. Export
    vertices_np = torch.nan_to_num(out_vertices_gpu.cpu(), nan=0.0).numpy().astype(np.float32)
    faces_np = out_faces_gpu.cpu().numpy().astype(np.int32)
    
    vertices_np[:, 1], vertices_np[:, 2] = vertices_np[:, 2], -vertices_np[:, 1]
    
    if faces_np.max() >= len(vertices_np):
        mask = faces_np.max(axis=1) < len(vertices_np)
        faces_np = faces_np[mask]
        
    colored_mesh = trimesh.Trimesh(
        vertices=vertices_np,
        faces=faces_np,
        process=False,
        vertex_colors=vertex_colors_rgba
    )
    
    print("DEBUG: Exporting sample.glb with Vertex Colors...")
    try:
        colored_mesh.export("sample.glb")
        print("DEBUG: GLB Export Success.")
    except Exception as e: print(f"GLB Export Failed: {e}")

    if use_tqdm: pbar.update(1); pbar.close()
    if verbose: print("Done")
    return colored_mesh
