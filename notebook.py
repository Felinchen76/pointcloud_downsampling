# Imports
from io import BytesIO
import time
import csv
import os
import open3d as o3d
import random
import requests
import tarfile
import numpy as np
from scipy.spatial import distance
from scipy.spatial.distance import pdist, squareform


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


def random_downsampling(model, endpoints):
    # get coordinates of the models
    coordinates = get_coordinates(model)
    # select random points for downsampling
    for i in range(len(coordinates) - endpoints):
        rannumb = random.randint(0, len(coordinates) - 1)
        del coordinates[rannumb]
    point_cloud = create_pointcloud_from_coordinates(coordinates)
    return point_cloud


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
        # Find point with largest min Distance
        min_mindist = np.min(eucl_distances, axis=0)
        # Find index of point with largest min DIstance
        max_min_distance_index = np.argmax(min_mindist)
        # add point that is farthest away
        retVal.append(coordinates[max_min_distance_index])
        # delete point from coordinates list 
        coordinates = np.delete(coordinates, max_min_distance_index, axis=0)
    return create_pointcloud_from_coordinates(np.array(retVal))


# built in function von open3d
def radius_outlier_removal_call(model):
    return model.remove_radius_outlier(nb_points=5, radius=0.05)


# add noise to pointcloud
def add_noise(model, noise_value):
    points = np.asarray(model.points)
    noise = np.random.normal(0, noise_value, size=points.shape)
    noisy_points = points + noise

    noisy_pc = o3d.geometry.PointCloud()
    noisy_pc.points = o3d.utility.Vector3dVector(noisy_points)
    return noisy_pc


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
    # convert voxelgrid to open3d Voxelgrid
    o3d_voxelgrid = o3d.geometry.VoxelGrid.create_from_point_cloud(input=model, voxel_size=voxel_size)
    # o3d.visualization.draw_geometries([o3d_voxelgrid])
    return o3d_voxelgrid


def voxel_filter(model, voxelgrid, voxel_size):
    # list where downsampled points will be saved
    downsampled_points = []
    # iterate over all voxel in the voxelgrid
    for voxel in voxelgrid.get_voxels():
        # get bounds of the voxel
        downsampled_points.extend(is_point_in_voxel(model, voxelgrid, voxel, voxel_size))
    downsampled_points = np.asarray(downsampled_points)
    return create_pointcloud_from_coordinates(downsampled_points)


def aggregate_points(points):
    # Aggregate the points by averaging, taking into account the z coordinate
    if len(points) == 0:
        return points
    aggregated_points = []
    aggregated_points.append(np.mean(points, axis=0))
    return aggregated_points


def is_point_in_voxel(model, voxelgrid, voxel, voxel_size):
    # get center point and see whether a point lies within the given distance/2 of the voxel size from the center
    voxel_center = voxelgrid.get_voxel_center_coordinate(voxel.grid_index)
    points_in_voxel = []
    half_size = voxel_size / 2.0
    # check, which points are lying within a voxel
    for point in model.points:
        if np.all(np.abs(point - voxel_center) <= half_size):
            points_in_voxel.append(point)
    points_in_voxel = aggregate_points(points_in_voxel)
    # print(points_in_voxel)
    return points_in_voxel


def create_points_from_voxel(voxel_model):
    # convert vector in numpy array
    vector_array = np.asarray(voxel_model)

    # create o3d point cloud
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(vector_array)

    return point_cloud


def point_cloud_to_ply(point_cloud, file_name):
    # safe downsampled point cloud as ply data
    file_name = "point_cloud_images/" + file_name + ".ply"
    if os.path.exists("point_cloud_images/" + file_name):
        os.remove(file_name)
    o3d.io.write_point_cloud(file_name, o3d.geometry.PointCloud(point_cloud.points))


def point_cloud_to_ply_simple(point_cloud, file_name):
    # safe downsampled point cloud as ply data
    file_name = "point_cloud_images_simple/" + file_name + ".ply"
    if os.path.exists("point_cloud_images_simple/" + file_name):
        os.remove(file_name)
    o3d.io.write_point_cloud(file_name, o3d.geometry.PointCloud(point_cloud.points))


# Laden der Punktewolken


cone = load_cad_model(r"cone.ply")
sphere = load_cad_model(r"sphere.ply")
cube = load_cad_model(r"cube.ply")
complex_cube = load_cad_model(r"complexCube.ply")
complex_cone = load_cad_model(r"hollowCone.ply")
complex_sphere = load_cad_model(r"complexSphere.ply")
pencil = load_cad_model(r"pencil_fein.ply")
# source: https://sketchfab.com/3d-models/davis-teapot-materialcleanup-547971eaf21d43f2b6cfcb6be0e7bf11
teapot = load_cad_model(r"teapot.ply")
# source: https://sketchfab.com/3d-models/book-ba04f5ac66194341bc7d437fb6c94674
book = load_cad_model(r"book.ply")


# ICP Algorithmus Implementierung


def icp_algorithm(source, target):
    # transform target point cloud
    transformation = np.array([[0.86, 0.5, 0.1, 0.5],
                               [-0.5, 0.86, 0.1, 0.99],
                               [0.0, -0.1, 0.99, 0.5],
                               [1.3, 0.0, 0.0, 1.0]])
    target = target.transform(transformation)

    threshold = 0.25  # max distance for deleting points
    initial_transformation = np.identity(4)  # initial guess of transformation

    # Open3D ICP Algorithmus
    reg_p2p = o3d.pipelines.registration.registration_icp(
        source, target, threshold, initial_transformation,
        o3d.pipelines.registration.TransformationEstimationPointToPoint())
    source.transform(reg_p2p.transformation)
    return reg_p2p


# Ergebnisse in CSV schreiben


def write_csv(array, filename):
    # Öffne die CSV-Datei im Schreibmodus
    with open(filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        for row in array:
            writer.writerow([row])


# Test Reproduzierbarkeit Rauschen und Gromov-Wasserstein Distanz


rd_wasserstein = []
vf_wasserstein = []
fp_wasserstein = []

num_iterations = 10
model = cone
array_noise = [0.05, 0.2, 1]

for noise in array_noise:
    for i in range(num_iterations):
        model = add_noise(model, noise)

        # random downsampling
        rd = random_downsampling(model, int(len(model.points) / 10 * 4))

        # voxelgrid
        vx_grid = create_voxel_grid(model, 0.2)
        vx = voxel_filter(model, vx_grid, 0.2)

        # farthest point downsampling
        fp = farthest_point_sampling(model, int(len(model.points) / 10 * 4))

        # Random Downsampling
        rd_wasserstein.append(gromov_wasserstein(rd, model))
        # Voxelgrid Filter
        vf_wasserstein.append(gromov_wasserstein(vx, model))
        # Farthest Point Downsampling
        fp_wasserstein.append(gromov_wasserstein(fp, model))

        print(i)

write_csv(rd_wasserstein, "rd_wasserstein.csv")
write_csv(vf_wasserstein, "vf_wasserstein.csv")
write_csv(fp_wasserstein, "fp_wasserstein.csv")

print(rd_wasserstein)
print(vf_wasserstein)
print(fp_wasserstein)

# #Results
# [0.2705866962525741, 0.09077863392603658, 0.09040394193458741, 0.09012493471875607, 0.09032806159254894, 0.09064595901392264, 0.09081015707767472, 0.09051769128054409, 0.09087100194533458, 0.09100855293880897, 0.0922270810223527, 0.09380357259387846, 0.09431286800254036, 0.09222332103671005, 0.09200205698689473, 0.09305364286790276, 0.09251122263758596, 0.09241460383976288, 0.09133464761685794, 0.09126492511066028, 0.09709291277717733, 0.10131523865305526, 0.11661935309986046, 0.12100764792659603, 0.11350478590567419, 0.11304301416130512, 0.11193416216748583, 0.11047181206616286, 0.10708674512802444, 0.1118358187180887]
# [5.427020676873916, 5.380804378791626, 5.313625265971279, 5.313716712643975, 5.185630644699037, 5.150867098484675, 5.068467151002501, 5.0530188164980006, 5.018578927648757, 4.934578556195397, 4.431708587827005, 4.0243785225533175, 3.6768758663047802, 3.385778716078735, 3.058387405848977, 2.80009206396991, 2.618203519934875, 2.3931745705221057, 2.2454185728055926, 2.0983466728254045, 0.6209216924309475, 0.3535481648429584, 0.1999622122093585, 0.15014320989310376, 0.10383583565177842, 0.09106672519463267, 0.07931363486179116, 0.06336485902402897, 0.05313326302929689, 0.059929603821885544]
# [6.217174431030691, 6.276125704909142, 6.303510295382989, 6.309374231854885, 6.3382631504508415, 6.354910421393487, 6.377586486379269, 6.386224815094574, 6.394788003141216, 6.396433269774092, 6.437916610428692, 6.488335342355508, 6.523835741561613, 6.560901838090065, 6.558417977852874, 6.563833674106713, 6.572971161296956, 6.597883712221651, 6.613228449136452, 6.621884489230054, 6.639423873671042, 6.708269979425678, 6.658113512070726, 6.687705779998461, 6.6114177090398805, 6.65606049537208, 6.712538377296238, 6.662146109845106, 6.7032918744320344, 6.5731617904265915]

# Laufzeittest


rd_times = []  # List of lists for random downsampling times
vf_times = []  # List of lists for voxel filter times
fp_times = []  # List of lists for farthest point sampling times

# Methode Spalte, Modell Zeile ?
# [][][]    
# [][][]
# model_array = [cube, sphere, cone, complex_cube, complex_cone, complex_sphere,
#               pencil, teapot, book]
model = cone
for round in range(num_iterations):
    # random downsampling
    start = time.time()
    rd = random_downsampling(model, int(len(model.points) / 10 * 4))
    end = time.time()
    elapsed_time = end - start
    rd_times.append(elapsed_time)

    # voxelgrid filter
    start = time.time()
    vx_grid = create_voxel_grid(model, 0.2)
    vx = voxel_filter(model, vx_grid, 0.7)
    end = time.time()
    elapsed_time = end - start
    vf_times.append(elapsed_time)

    # farthest point downsampling
    start = time.time()
    fp = farthest_point_sampling(model, int(len(model.points) / 10 * 4))
    end = time.time()
    fp_pc = o3d.geometry.PointCloud()
    fp_pc.points = o3d.utility.Vector3dVector(np.asarray(fp.points))
    elapsed_time = end - start
    fp_times.append(elapsed_time)  # Add the time to the corresponding model's list

write_csv(rd_times, "rd_times.csv")
write_csv(vf_times, "vf_times.csv")
write_csv(fp_times, "fp_times.csv")

# ICP Tests


# Vergleich auf den Originalwolken
original_fitness = []
original_inlier = []

for model in model_array:
    # ICP auf den Originalpunktewolken
    rd_icp = icp_algorithm(add_noise(model, 0.7), model)
    original_fitness.append(rd_icp.fitness)
    original_inlier.append(rd_icp.inlier_rmse)

write_csv(original_fitness, "original_fitness.csv")
write_csv(original_fitness, "original_icp_fitness.csv")

rd_icp_fitness = []
rd_icp_inlier = []
vf_icp_fitness = []
vf_icp_inlier = []
fp_icp_fitness = []
fp_icp_inlier = []

for model in model_array:
    # random downsampling
    model_rd = random_downsampling(model, int(len(model.points) / 10 * 4))
    model_rd_pc = o3d.geometry.PointCloud()
    model_rd_pc.points = o3d.utility.Vector3dVector(np.asarray(model_rd.points))
    rd_icp = icp_algorithm(model_rd_pc, model)
    rd_icp_fitness.append(rd_icp.fitness)
    rd_icp_inlier.append(rd_icp.inlier_rmse)

    # voxel grid filter
    model_voxel_grid = create_voxel_grid(model, 0.2)
    model_voxel = voxel_filter(model, model_voxel_grid, 0.2)
    vf_icp = icp_algorithm(model_voxel, model)
    vf_icp_fitness.append(vf_icp.fitness)
    vf_icp_inlier.append(rd_icp.inlier_rmse)

    # farthest point downsampling#
    model_fp = farthest_point_sampling(model, int(len(model.points) / 10 * 4))
    model_fp_pc = o3d.geometry.PointCloud()
    model_fp_pc.points = o3d.utility.Vector3dVector(np.asarray(model_fp.points))
    fp_icp = icp_algorithm(model_fp_pc, model)
    fp_icp_fitness.append(fp_icp.fitness)
    fp_icp_inlier.append(fp_icp.inlier_rmse)

write_csv(rd_icp_fitness, "rd_icp_fitness.csv")
write_csv(vf_icp_fitness, "vf_icp_fitness.csv")
write_csv(fp_icp_fitness, "fp_icp_fitness.csv")
write_csv(rd_icp_inlier, "rd_icp_inlier.csv")
write_csv(vf_icp_inlier, "vf_icp_inlier.csv")
write_csv(fp_icp_inlier, "fp_icp_inlier.csv")

# Noisiness test basic models


model_array = [cube, cone, sphere]
model_names = ["cube", "cone", "sphere"]
for index, model in enumerate(model_array):
    # create noisy pointclouds
    noisy_model = add_noise(model, 0.1)

    # Random Downsampling
    rd_noisy = random_downsampling(noisy_model, int(len(noisy_model.points) / 10 * 4))
    point_cloud_to_ply(rd_noisy, "noisy_rd_" + model_names[index - 1])

    # Voxel Grid Filter
    noisy_model_grid = create_voxel_grid(noisy_model, 0.2)
    noisy_model_voxel_pc = voxel_filter(noisy_model, noisy_model_grid, 0.2)
    point_cloud_to_ply(noisy_model_voxel_pc, "noisy_vf_" + model_names[index - 1])

    # Farthest Point Downsampling
    noisy_model_fp = farthest_point_sampling(noisy_model, int(len(noisy_model.points) / 10 * 4))
    noisy_model_fp_pc = o3d.geometry.PointCloud()
    noisy_model_fp_pc.points = o3d.utility.Vector3dVector(np.asarray(noisy_model_fp.points))
    point_cloud_to_ply(noisy_model_fp_pc, "noisy_fp_" + model_names[index - 1])

noisetest = load_cad_model(r"point_cloud_images/noisy_rd_complex_sphere.ply")
visualize_model(noisetest)

# noisiness tests complex models


model_array = [complex_sphere, complex_cube, complex_cone]
model_names = ["complex_sphere", "complex_cube", "complex_cone"]

for index, complex_model in enumerate(model_array):
    noisy_complex_model = add_noise(complex_model, 1.3)
    rd_complex_noisy = random_downsampling(noisy_complex_model, int(len(noisy_complex_model.points) / 10 * 4))
    point_cloud_to_ply(rd_complex_noisy, "noisy_rd_" + model_names[index])

    noisy_complex_model_grid = create_voxel_grid(noisy_complex_model, 0.2)
    noisy_complex_model_voxel_pc = voxel_filter(noisy_complex_model, noisy_complex_model_grid, 0.2)
    point_cloud_to_ply(noisy_complex_model_voxel_pc, "noisy_vf_" + model_names[index])

    noisy_complex_model_fp = farthest_point_sampling(noisy_complex_model, int(len(noisy_complex_model.points) / 10 * 4))
    noisy_complex_model_fp_pc = o3d.geometry.PointCloud()
    noisy_complex_model_fp_pc.points = o3d.utility.Vector3dVector(np.asarray(noisy_complex_model_fp.points))
    point_cloud_to_ply(noisy_complex_model_fp_pc, "noisy_fp_" + model_names[index])

# noisiness tests objects


model_array = [book, teapot, pencil]
model_names = ["book", "teapot", "pencil"]

for index, model_object in enumerate(model_array):
    # random downsampling
    noisy_model_object = add_noise(model_object, 0.1)
    rd_noisy = random_downsampling(noisy_model_object, int(len(noisy_model_object.points) / 10 * 4))
    point_cloud_to_ply(rd_noisy, "noisy_rd_" + model_names[index])

    # voxel grid filter
    noisy_model_object_grid = create_voxel_grid(noisy_model_object, 0.2)
    noisy_object_voxel_pc = voxel_filter(noisy_model_object, noisy_model_object_grid, 0.2)
    point_cloud_to_ply(noisy_object_voxel_pc, "noisy_vf_" + model_names[index])

    # farthest point downsampling
    noisy_model_object_fp = farthest_point_sampling(noisy_model_object, int(len(noisy_model_object.points) / 10 * 4))
    noisy_model_object_fp_pc = o3d.geometry.PointCloud()
    noisy_model_object_fp_pc.points = o3d.utility.Vector3dVector(np.asarray(noisy_model_object_fp.points))
    point_cloud_to_ply(noisy_model_object_fp_pc, "noisy_fp_" + model_names[index])

visualize_model(load_cad_model(r"point_cloud_images/noisy_rd_cube.ply"))

for point in cone.points:
    print(point)

# DOWNSAMPLING SIMPLE AUFBAU


model_array = [cube, cone, sphere]
model_names = ["cube", "cone", "sphere"]
for index, model in enumerate(model_array):
    # create noisy pointclouds
    noisy_model = add_noise(model, 0.1)

    # Random Downsampling
    rd_noisy = random_downsampling(noisy_model, int(len(noisy_model.points) / 10 * 2))
    point_cloud_to_ply_simple(rd_noisy, "rd_" + model_names[index - 1])

    # Voxel Grid Filter
    noisy_model_grid = create_voxel_grid(noisy_model, 0.6)
    noisy_model_voxel_pc = voxel_filter(noisy_model, noisy_model_grid, 0.6)
    point_cloud_to_ply_simple(noisy_model_voxel_pc, "vf_" + model_names[index - 1])

    # Farthest Point Downsampling
    noisy_model_fp = farthest_point_sampling(noisy_model, int(len(noisy_model.points) / 10 * 2))
    noisy_model_fp_pc = o3d.geometry.PointCloud()
    noisy_model_fp_pc.points = o3d.utility.Vector3dVector(np.asarray(noisy_model_fp.points))
    point_cloud_to_ply_simple(noisy_model_fp_pc, "fp_" + model_names[index - 1])
