from io import BytesIO
import os
import open3d as o3d
import random
import requests
import tarfile
import numpy as np
from scipy.spatial import distance
from sklearn.metrics import mean_squared_error
from skopt.space import Real

from scipy.spatial.distance import cdist, pdist, squareform

"""
Shape of the point clouds:
pc1.shape = (NumberOfPoints1, Dimension)
pc2.shape = (NumberOfPoints2, Dimension)
"""


def gromov_wasserstein(pc1: np.ndarray, pc2: np.ndarray) -> float:
    def dist_ecc_fast(ecc, u):
        return (np.mean(ecc <= u))

    out = 0
    # Konvertiere die Punktwolken in NumPy-Arrays
    pc1 = np.asarray(pc1.points)
    pc2 = np.asarray(pc2.points)

    # Reshape input matrices if necessary
    if pc1.ndim == 1:
        pc1 = pc1.reshape(-1, 1)
    if pc2.ndim == 1:
        pc2 = pc2.reshape(-1, 1)

    ecc1 = squareform(pdist(pc1)).mean(0)
    ecc2 = squareform(pdist(pc2)).mean(0)
    unique_ecc = np.unique(np.concatenate((ecc1, ecc2)))
    for i in range(unique_ecc.shape[0] - 1):
        u = unique_ecc[i]
        out += (unique_ecc[i + 1] - unique_ecc[i]) * np.abs(dist_ecc_fast(ecc1, u) - dist_ecc_fast(ecc2, u))

    return (0.5 * out)


def chamfer_distance(pc1: np.ndarray, pc2: np.ndarray) -> float:
    dist = cdist(pc1, pc2)
    ch_dist = (np.min(dist, axis=1).mean() + np.min(dist, axis=0).mean()) / 2
    return ch_dist


def average_ratio(pc1: np.ndarray, pc2: np.ndarray, Dist_list: list) -> float:
    d = cdist(pc1, pc2)
    d0 = d.min(0)
    d1 = d.min(1)

    avr = 0
    for i, dist in enumerate(Dist_list):
        avr += (i + 1) * ((d1 <= dist).sum() / pc1.shape[0] + (d0 <= dist).sum() / pc2.shape[0])
    return avr / (len(Dist_list) ** 2 + len(Dist_list))


def ratio(pc1: np.ndarray, pc2: np.ndarray, thr_list=np.array([0.1, 0.5, 1, 2, 4])) -> list:
    d = np.min(cdist(pc1, pc2), axis=1)
    r = (d <= thr_list[:, None]).mean(1)
    return thr_list, r

def create_cone_pcd(radius, height, numberofpoints):
    """
    Returns a point cloud that represents a cylinder
    :param radius: radius of the cylinder
    :param height: height of the cylinder
    :return: point_cloud (open3d.geometry.PointCloud): die erstellte Punktewolke
    """
    # http://www.open3d.org/docs/release/tutorial/geometry/mesh.html#Sampling
    conemesh = o3d.geometry.TriangleMesh.create_cone(radius, height, resolution=20, split=1)
    # calculate the vertex normal for the cone
    conemesh.compute_vertex_normals()
    o3d.visualization.draw_geometries([conemesh])
    # distribute dots evenly on the surface
    pcd = conemesh.sample_points_uniformly(numberofpoints)
    return pcd


# Funktion zum Erstellen einer Punktwolke eines Kegels
def create_hollow_cone(big_cone, small_cone):
    mesh_big = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(big_cone)
    mesh_small = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(small_cone)
    small_cone.translate([0, 0, 1.5])
    hollow_cone = mesh_big + mesh_small
    return hollow_cone


def create_pcd_from_mesh(mesh):
    mesh.compute_vertex_normals()
    o3d.visualization.draw_geometries([mesh])
    # distribute dots evenly on the surface
    return mesh.sample_points_uniformly(500)


def load_model(link, path):
    # http://ycb-benchmarks.s3-website-us-east-1.amazonaws.com/
    response = requests.get(link)
    tgz_data = BytesIO(response.content)
    # set the current working directory to the script's directory
    script_directory = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_directory)
    with tarfile.open(fileobj=tgz_data, mode="r:gz") as tar_ref:
        tar_ref.extractall(script_directory)
    # join paths
    model_path = os.path.join(script_directory, path, "clouds", "merged_cloud.ply")
    # load pointcloud
    pcd = o3d.io.read_point_cloud(model_path)
    return pcd


def load_cad_model(model):
    # load model generated in freecad
    return o3d.io.read_point_cloud(model)


def visualize_model(model):
    o3d.visualization.draw_geometries([model])


def get_num_points(model):
    print(len(model.points))


def create_pointcloud_from_coordinates(coordinates):
    # create point cloud with coordinates
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(coordinates)
    return pcd


def get_coordinates(model):
    coordinates = [list(point) for point in model.points]
    # print(coordinates[:50])
    return coordinates


# random downsampling
def random_downsampling(model, endpoints):
    # get coordinates of the model
    coordinates = get_coordinates(model)
    # select random points for downsampling
    for i in range(len(coordinates) - endpoints):
        rannumb = random.randint(0, len(coordinates) - 1)
        del coordinates[rannumb]
    return coordinates


def farthest_point_sampling1(model, num_points_keep):
    coordinates = np.array(get_coordinates(model))
    retVal = []
    # to make runs comparable
    random.seed(13)
    # generate "random" int
    randint = random.randint(0, len(coordinates) - 1)
    # select random point from model
    retVal = np.append(retVal, coordinates[randint])
    # delete chosen point from original model after it was added to the downsampled cloud
    coordinates = np.delete(coordinates, randint, axis=0)
    while len(retVal) < num_points_keep:
        retValnp = np.array(retVal)
        # berechne die Entfernungen der ausgewählten Punkte zu den ausgewählten Punkten
        eucl_distances = distance.cdist(retValnp, coordinates, 'euclidean')
        # finde den weitesten Punkt heraus
        # füge den am weitesten entfernten Punkt der Liste hinzu
        retVal = np.append(retValnp, np.max(eucl_distances, axis=0))
        # punkt aus der Liste mit den ursprünglichen Koordinaten löschen
        min_distance_index = np.argmax(eucl_distances)
        coordinates = np.delete(coordinates, min_distance_index, axis=0)
    return retVal


def farthest_point_sampling(model, num_points_keep):
    coordinates = np.array(get_coordinates(model))
    retVal = []
    # to make runs comparable
    random.seed(13)
    # generate "random" int
    randint = random.randint(0, len(coordinates) - 1)
    # select random point from model
    retVal.append(coordinates[randint])
    # delete chosen point from original model after it was added to the downsampled cloud
    coordinates = np.delete(coordinates, randint, axis=0)
    while len(retVal) < num_points_keep:
        # Berechne die euklidischen Distanzen der ausgewählten Punkte zu den verbleibenden Punkten
        eucl_distances = distance.cdist(retVal, coordinates, 'euclidean')
        # Finden Sie den Punkt mit der größten Mindestdistanz
        min_mindist = np.min(eucl_distances, axis=0)
        # Finden Sie den Index des Punktes mit der größten Mindestdistanz
        max_min_distance_index = np.argmax(min_mindist)
        # Fügen Sie den am weitesten entfernten Punkt der Liste hinzu
        retVal.append(coordinates[max_min_distance_index])
        # Entfernen Sie den ausgewählten Punkt aus den verbleibenden Koordinaten
        coordinates = np.delete(coordinates, max_min_distance_index, axis=0)
    return np.array(retVal)


# built in function von open3d?
def radius_outlier_removal_call(model):
    return model.remove_radiues_outlier(nb_points=5, radius=0.05)


# voxel downsampling
def voxel_downsampling(model):
    pcd = model.voxel_down_sample(voxel_size=0.05)
    return pcd


# with median -> aggregation method as parameter??
def create_voxel_grid(model, voxel_size):
    model_points = np.array(get_coordinates(model))
    min_bound = np.min(model_points, axis=0)
    max_bound = np.max(model_points, axis=0)

    dimensions = np.ceil((max_bound - min_bound) / voxel_size).astype(int)

    voxelgrid = np.zeros(dimensions)

    for point in model_points:
        voxel_coordinates = ((point - min_bound) / voxel_size).astype(int)
        # -1 needed in order to avoid index out of bounds
        voxelgrid[tuple(voxel_coordinates - 1)] += 1
    # visualize voxel grid to check if its correct
    # convert voxelgrid to open3d Voxelgrid
    o3d_voxelgrid = o3d.geometry.VoxelGrid.create_from_point_cloud(input=model, voxel_size=voxel_size)
    o3d.visualization.draw_geometries([o3d_voxelgrid])
    return o3d_voxelgrid


def voxel_filter(model, voxelgrid, voxel_size):
    # list where downsampled points will be saved
    downsampled_points = []
    # iterate over all voxel in the voxelgrid
    for voxel in voxelgrid.get_voxels():
        # get bounds of the voxel
        downsampled_points.extend(is_point_in_voxel(model, voxelgrid, voxel, voxel_size))
    downsampled_points = np.asarray(downsampled_points)
    return o3d.utility.Vector3dVector(downsampled_points)


def aggregate_points(points):
    # Aggregate the points by averaging, taking into account the z coordinate
    if len(points) == 0:
        return points
    aggregated_points = []
    aggregated_points.append(np.mean(points, axis=0))
    return aggregated_points


def is_point_in_voxel(model, voxelgrid, voxel, voxel_size):
    # old idea: get the 8 bounding points of a voxel
    # voxel_points = np.asarray(voxelgrid.get_voxel_bounding_points(voxel.grid_index))

    # get center point and see whether a point lies within the given distance/2 of the voxel size from the center
    voxel_center = voxelgrid.get_voxel_center_coordinate(voxel.grid_index)
    points_in_voxel = []
    half_size = voxel_size / 2.0
    # Überprüfe, welche Punkte innerhalb des Voxels liegen
    for point in model.points:
        if np.all(np.abs(point - voxel_center) <= half_size):
            points_in_voxel.append(point)
    points_in_voxel = aggregate_points(points_in_voxel)
    # print(points_in_voxel)
    return points_in_voxel


# Idea: Compare the convex hull of the original pointcloud
# with the convex hull of the downsampled point cloud to assess the quality of the downsampling method.
def get_convex_hull(model):
    hull, _ = model.compute_convex_hull()
    return hull


def calc_vol_of_hull(hull):
    # https://gist.github.com/JoseLlorensRipolles/fd3faa766527f1a3699eb58c985a30c0
    hullVol = hull.get_volume()
    print(hullVol)

try:
    print("hi")
    # pcd = load_model()
    # visualize_model(pcd)
    # get_num_points(pcd)
    # coordinates = random_downsampling(pcd, 70000)
    # rndPtCloud = create_pointcloud_from_coordinates(coordinates)
    # visualize_model(rndPtCloud)
except Exception:
    print("an error has occured loading the data")


from skopt import BayesSearchCV, gp_minimize


def bayesian_optimization(space, ds_score, n_calls):
    result = gp_minimize(ds_score, space, n_calls=n_calls, random_state=42)
    return result


import torch
from chamferdist import ChamferDistance

def compute_chamfer_dist(original_pcd, downsampled_pcd):
    pcd_o = np.asarray(original_pcd.points)
    pcd_d = np.asarray(downsampled_pcd.points)
    # Konvertiere die numpy-Array in einen PyTorch Tensor
    pcd1_tensor = torch.tensor(pcd_o, dtype=torch.float32)
    pcd2_tensor = torch.tensor(pcd_d, dtype=torch.float32)
    chamferDist = ChamferDistance()
    #wofür benötigt man vorwärts und rückwärtsdistanz?
    dist_forward = chamferDist(pcd1_tensor.unsqueeze(0), pcd2_tensor.unsqueeze(0))
    print("Vorwärtsdistanz:", dist_forward.detach().cpu().item())
    dist_backwards = chamferDist(pcd2_tensor.unsqueeze(0), pcd1_tensor.unsqueeze(0))
    print("Vorwärtsdistanz:", dist_backwards.detach().cpu().item())
    avg_dist_forward = torch.mean(dist_forward)  # oder torch.sum(dist_forward)
    return avg_dist_forward.item()

# load the 3 chosen YCB Objects
#ycb_round = load_model("http://ycb-benchmarks.s3-website-us-east-1.amazonaws.com/data/berkeley/049_small_clamp/049_small_clamp_berkeley_meshes.tgz","049_small_clamp")
#visualize_model(ycb_round)
#ycb_cube = load_model("http://ycb-benchmarks.s3-website-us-east-1.amazonaws.com/data/berkeley/065-a_cups/065-a_cups_berkeley_meshes.tgz","065-a_cups")
#visualize_model(ycb_cube)
#ycb_cone = load_model("http://ycb-benchmarks.s3-website-us-east-1.amazonaws.com/data/berkeley/012_strawberry/012_strawberry_berkeley_meshes.tgz","012_strawberry")
#visualize_model(ycb_cone)
# creat synthetic big cone for tests
#conepcd_big = create_cone_pcd(1, 4, 500)

# to do: pfad so angeben, dass man das immer automatisch im projekt hat!

cone = load_cad_model(r"C:\Users\lockf\downsampling_project\cone.ply")
visualize_model(cone)

sphere = load_cad_model(r"C:\Users\lockf\downsampling_project\sphere.ply")
visualize_model(sphere)

cube = load_cad_model(r"C:\Users\lockf\downsampling_project\cube.ply")
visualize_model(cube)

hollow_cone = load_cad_model(r"C:\Users\lockf\downsampling_project\hollowCone.ply")
visualize_model(hollow_cone)

# To Do
# complex sphere
complex_sphere = load_cad_model(r"C:\Users\lockf\downsampling_project\complexSphere.ply")
visualize_model(complex_sphere)

# To Do
# complex cube
pencil = load_cad_model(r"C:\Users\lockf\downsampling_project\pencil_fein.ply")
visualize_model(pencil)

#To Do
# teapod
# source: https://sketchfab.com/3d-models/davis-teapot-materialcleanup-547971eaf21d43f2b6cfcb6be0e7bf11
teapot = load_cad_model(r"C:\Users\lockf\downsampling_project\teapot.ply")
visualize_model(teapot)

#To Do
# book
# source: https://sketchfab.com/3d-models/book-ba04f5ac66194341bc7d437fb6c94674
book = load_cad_model(r"C:\Users\lockf\downsampling_project\book.ply")
visualize_model(book)

# downsample the cone via random downsampling
coordinatesConeDownsampled = random_downsampling(hollow_cone, 250)
coneDownsampledR = create_pointcloud_from_coordinates(coordinatesConeDownsampled)

# Visualize cone
visualize_model(hollow_cone)
visualize_model(coneDownsampledR)

# downsample via voxel filter
voxel_size = 0.2
voxelgrid = create_voxel_grid(hollow_cone, voxel_size)
coneDownsampledV = create_pointcloud_from_coordinates(voxel_filter(hollow_cone, voxelgrid, voxel_size))



#parameter optimization for voxelgrid filter

# Define the parameter space for optimization
space = [
    Real(0.1, 1.0, name='voxelsize'),
    # Add more parameters as needed
]

score = compute_chamfer_dist(hollow_cone, coneDownsampledV)
result = bayesian_optimization(space, lambda params: compute_chamfer_dist(hollow_cone, voxel_downsampling(hollow_cone)), 20)
voxel_size = result.x[0]
voxelgrid2 = create_voxel_grid(hollow_cone, voxel_size)
compute_chamfer_dist(hollow_cone, coneDownsampledV)


# calculate convex hulls
hullOriginal = get_convex_hull(hollow_cone)
hullR = get_convex_hull(coneDownsampledR)
hullV = get_convex_hull(coneDownsampledV)

# compare volumes of the convex hulls
# hullVolOriginal = calc_vol_of_hull(hullOriginal)
# hullVolR = calc_vol_of_hull(hullR)
# hullVOlV = calc_vol_of_hull(hullV)

# downsample fps
cone_downsampledF = farthest_point_sampling(hollow_cone, 250)
# wird benötigt, weil ich das numpy array konvertieren muss
point_cloud = o3d.geometry.PointCloud()
point_cloud.points = o3d.utility.Vector3dVector(cone_downsampledF)
visualize_model(point_cloud)

# radius outlier removal
#to do


print(gromov_wasserstein(hollow_cone, coneDownsampledR))


