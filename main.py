from io import BytesIO
import os
import open3d as o3d
import random
import requests
import tarfile
import numpy as np

def create_cone_pcd(radius, height, numberofpoints):
    """
    Returns a point cloud that represents a cylinder
    :param radius: radius of the cylinder
    :param height: height of the cylinder
    :return: point_cloud (open3d.geometry.PointCloud): die erstellte Punktewolke
    """
    #http://www.open3d.org/docs/release/tutorial/geometry/mesh.html#Sampling
    conemesh = o3d.geometry.TriangleMesh.create_cone(radius, height, resolution=20, split=1)
    # calculate the vertex normal for the cone
    conemesh.compute_vertex_normals()
    o3d.visualization.draw_geometries([conemesh])
    # distribute dots evenly on the surface
    pcd = conemesh.sample_points_uniformly(numberofpoints)
    return pcd

def load_model ():
    # http://ycb-benchmarks.s3-website-us-east-1.amazonaws.com/
    model_url = "http://ycb-benchmarks.s3-website-us-east-1.amazonaws.com/data/berkeley/001_chips_can/001_chips_can_berkeley_meshes.tgz"
    response = requests.get(model_url)
    tgz_data = BytesIO(response.content)
    # set the current working directory to the script's directory
    script_directory = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_directory)
    with tarfile.open(fileobj=tgz_data, mode="r:gz") as tar_ref:
        tar_ref.extractall(script_directory)
    # join paths
    model_path = os.path.join(script_directory, "001_chips_can","clouds","merged_cloud.ply")
    # load pointcloud
    pcd = o3d.io.read_point_cloud(model_path)
    return pcd


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
    for i in range(len(coordinates)-endpoints):
        rannumb = random.randint(0, len(coordinates)-1)
        del coordinates[rannumb]
    return coordinates

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
        voxelgrid[tuple(voxel_coordinates)] += 1
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
    #Aggregate the points by averaging, taking into account the z coordinate
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
    #print(points_in_voxel)
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
    pcd = load_model()
    visualize_model(pcd)
    get_num_points(pcd)
    #coordinates = random_downsampling(pcd, 70000)
    #rndPtCloud = create_pointcloud_from_coordinates(coordinates)
    #visualize_model(rndPtCloud)
except Exception:
    print("an error has occured loading the data")

# creat synthetic cone for tests
conepcd = create_cone_pcd(1,4,500)

# downsample the cone via random downsampling
coordinatesConeDownsampled = random_downsampling(conepcd, 250)
coneDownsampledR = create_pointcloud_from_coordinates(coordinatesConeDownsampled)

# Visualize cone
visualize_model(conepcd)
visualize_model(coneDownsampledR)

# downsample via voxel filter
voxel_size = 0.2
voxelgrid = create_voxel_grid(conepcd, voxel_size)
coneDownsampledV = create_pointcloud_from_coordinates(voxel_filter(conepcd, voxelgrid, voxel_size))

# calculate convex hulls
hullOriginal = get_convex_hull(conepcd)
hullR = get_convex_hull(coneDownsampledR)
hullV = get_convex_hull(coneDownsampledV)

# compare volumes of the convex hulls
hullVolOriginal = calc_vol_of_hull(hullOriginal)
hullVolR = calc_vol_of_hull(hullR)
hullVOlV = calc_vol_of_hull(hullV)
