from pathlib import Path

import numpy as np

from lightning.assets.vista.multi_plot import plot_mesh_montage, add_mesh, plotter

from lightning.assets.vista.plot_util.container import stringify
from lightning.assets.vista.plot_util.execution import busy_wait
# import pygeodesic.geodesic as geodesic
from plyfile import PlyData, PlyElement
import meshio
# import meshplex
from plyfile import PlyData, PlyElement
from tqdm import tqdm
import os
from scipy.spatial.distance import directed_hausdorff

# TODO - insert explicit mesh naming, so we do not relay on updating the last mesh.
# ---------------------------------------------------------------------------------------------------------------------#
#
# ---------------------------------------------------------------------------------------------------------------------#

def multianimate(vss, fs=None,gif_name=None, titles=None, color='lightcoral', **plt_args):
    print('Orient the view, then press "q" to start animation')
    p, ms = plot_mesh_montage(vs=[vs[0] for vs in vss], fs=fs, titles=titles, auto_close=False, colors=color,
                              ret_meshes=True, **plt_args)
    if gif_name is not None:
        p.open_gif(gif_name)
    num_frames_per_vs = [len(vs) for vs in vss]
    longest_sequence = max(num_frames_per_vs)
    for i in range(longest_sequence):  # Iterate over all frames
        for mi, m in enumerate(ms):
            if i < num_frames_per_vs[mi]:
                p.update_coordinates(points=vss[mi][i], mesh=m, render=False)
                if gif_name is not None:
                    p.write_frame()
                if i == num_frames_per_vs[mi] - 1:
                    for k, actor in p.renderers[mi]._actors.items():
                        if k.startswith('PolyData'):
                            actor.GetProperty().SetColor([0.8] * 3)  # HACKY!
        for renderer in p.renderers:  # TODO - is there something smarter than this?
            renderer.ResetCameraClippingRange()
        p.update()
    p.show(full_screen=False)  # To allow for screen hanging.

# ---------------------------------------------------------------------------------------------------------------------#
#
# ---------------------------------------------------------------------------------------------------------------------#


def animate(vs, f=None, gif_name=None, titles=None, first_frame_index=0, pause=0.05, color='w',
            callback_func=None, setup_func=None, **plt_args):
    p = plotter()
    add_mesh(p, vs[first_frame_index], f, color=color, as_a_single_mesh=True, **plt_args)
    if setup_func is not None:
        setup_func(p)
    # Plot first frame. Normals are not supported
    print('Orient the view, then press "q" to start animation')
    if titles is not None:
        titles = stringify(titles)
        p.add_text(titles[first_frame_index], position='upper_edge', name='my_title')
    p.show(auto_close=False, full_screen=True)

    # Open a gif
    if gif_name is not None:
        p.open_gif(gif_name)

    for i, v in enumerate(vs):
        if titles is not None:
            p.add_text(titles[i], position='upper_edge', name='my_title')
        if callback_func is not None:
            v = callback_func(p, v, i, len(vs))
        p.update_coordinates(v, render=False)
        # p.reset_camera_clipping_range()
        if pause > 0:
            busy_wait(pause)  # Sleeps crashes the program

        if gif_name is not None:
            p.write_frame()

        if i == len(vs) - 1:
            for k, actor in p.renderer._actors.items():
                actor.GetProperty().SetColor([0.8] * 3)  # HACKY!

        p.update()
    p.show(full_screen=False)  # To allow for screen hanging.


def animate_color(colors, v, f=None, gif_name=None, titles=None, first_frame_index=0, pause=0.05, setup_func=None,
                  reset_clim=False, **plt_args):
    # TODO - find some smart way to merge to animate
    # TODO - fix jitter in the title change
    p = plotter()
    add_mesh(p, v, f, color=colors[first_frame_index], as_a_single_mesh=True, **plt_args)
    if setup_func is not None:
        setup_func(p)
    # Plot first frame. Normals are not supported
    print('Orient the view, then press "q" to start animation')
    if titles is not None:
        titles = stringify(titles)
        p.add_text(titles[first_frame_index], position='upper_edge', name='my_title')
    p.show(auto_close=False, full_screen=True)

    # Open a gif
    if gif_name is not None:
        p.open_gif(gif_name)

    for i, color in enumerate(colors):
        p.update_scalars(color, render=False)
        if reset_clim:
            p.update_scalar_bar_range(clim=[np.min(color), np.max(color)])
        if titles is not None:
            p.add_text(titles[i], position='upper_edge', name='my_title')
        # p.reset_camera_clipping_range()
        if pause > 0:
            busy_wait(pause)  # Sleeps crashes the program

        if gif_name is not None:
            p.write_frame()

        if i == len(colors) - 1:
            for k, actor in p.renderer._actors.items():
                actor.GetProperty().SetColor([0.8] * 3)  # HACKY!

        p.update()
    p.show(full_screen=False)  # To allow for screen hanging

def load_off(file_path):
    file = meshio.read(file_path, "off")
    v, f = file.points, file.get_cells_type("triangle")
    v = np.array([list(vertex) for vertex in v])
    f = np.array(list([list(face) for face in f])).squeeze()
    #v, f = np.array(v, f)
    return v, f
    #mesh = meshplex.MeshTri(np.array(v), np.array(f))
    #return mesh

def load_ply(file_path):
    dfaust_path = r"C:\Users\ido.iGIP1\hy\Ramp\shape_completion-main\src\core\results\debug_experiment\version_19\completions\DFaustProj"
    file = PlyData.read(dfaust_path+"\\"+file_path)
    v, f = file["vertex"].data, file["face"].data
    v = np.array([list(vertex) for vertex in v])
    f = np.array(list([list(face) for face in f])).squeeze() 
    return v, f
    #mesh = meshplex.MeshTri(np.array(v), f)
    #return mesh

def calculate_volume_of_off(file_path):
    file = meshio.read(file_path, "off")
    # print(mesh)
    # print(file.__dict__)
    v, f = file.points, file.get_cells_type("triangle")
    mesh = meshplex.MeshTri(np.array(v), np.array(f))
    V = np.sum(mesh.cell_volumes)
    return V

def calculate_volume_of_ply(file_path):
    dfaust_path = r"C:\Users\ido.iGIP1\hy\Ramp\shape_completion-main\src\core\results\debug_experiment\version_19\completions\DFaustProj"
    file = PlyData.read(dfaust_path+"\\"+file_path)
    v, f = file["vertex"].data, file["face"].data
    v = np.array([list(vertex) for vertex in v])
    f = np.array(list([list(face) for face in f])).squeeze() 
    mesh = meshplex.MeshTri(np.array(v), f)
    V = np.sum(mesh.cell_volumes)
    return V

def find_minima(array, value,metric):
    array = np.asarray(array)
    if(metric == "hausdorff" or metric == "L2"):  
        value=0
    idx = (np.abs(array - value)).argmin()
    return idx

def create_best_animation_by_volume(subject,pose):
    from os import listdir
    from os.path import isfile, join
    dfaust_path = r"C:\Users\ido.iGIP1\hy\Ramp\shape_completion-main\src\core\results\debug_experiment\version_19\completions\DFaustProj"
    onlyfiles = [f for f in listdir(dfaust_path) if isfile(join(dfaust_path, f)) and f.startswith(f"gt_{subject}") and f.endswith("res.ply")]
    files_and_digits = [(f,[int(s) for s in f.split("_") if s.isdigit()][-1]) for f in onlyfiles]
    poses = {}
    for (f,digit) in files_and_digits:
        if digit not in poses:
            poses[digit]=[]
        poses[digit]+=[f]
    file_names = [(r"R:\Mano\data\DFaust\DFaust\full\50007\jiggle_on_toes\{0:05d}.OFF".format(d),d) for d in range(285)]
    volumes_gt = [calculate_volume_of_off(f) for f,d in file_names]
    best_reconstruction_by_volume = {}
    for (id,file_paths) in tqdm(poses.items()):
        volumes = np.array([calculate_volume_of_ply(path) for path in file_paths])
        idx = find_nearest(volumes,volumes_gt[id])
        best_reconstruction_by_volume[id]=file_paths[int(idx)]
    anim_plys = list(best_reconstruction_by_volume.values())
    # plydata = PlyData.read(dfaust_path)
    plys = [PlyData.read(dfaust_path + "\\" +f) for f in anim_plys]
    faces = np.array(list([list(face) for face in plys[0]["face"].data])).squeeze() 
    # np.array(list([list(vertex) for vertex in vertices]))
    vs = [np.array(list([list(vertex) for vertex in ply["vertex"].data])) for ply in plys]
    animate(vs,faces,gif_name=r"C:\Users\ido.iGIP1\hy\Ramp\animate_{}_{}.gif".format(subject, pose))

def calculate_metric(v, f, metric,v_gt=None,f_gt=None):
    dfaust_path = r"C:\Users\ido.iGIP1\hy\Ramp\shape_completion-main\src\core\results\debug_experiment\version_19\completions\DFaustProj"
    # file = PlyData.read(dfaust_path+"\\"+file_path)
    # v, f = file["vertex"].data, file["face"].data
    # v = np.array([list(vertex) for vertex in v])
    # f = np.array(list([list(face) for face in f])).squeeze() 
    # mesh = meshplex.MeshTri(np.array(v), f)
    if (metric == 'volume'):
        mesh = meshplex.MeshTri(np.array(v), f)
        res = np.sum(mesh.cell_volumes)
    elif (metric == 'geodesic'):
        geoalg=geodesic.PyGeodesicAlgorithmExact(v,f)
        source_indices = np.array([0])
        target_indices = None
        distances, best_source = geoalg.geodesicDistances(source_indices, target_indices)
        res = np.mean(distances) 
    elif (metric == "hausdorff"):
        #assert(v_gt is not None,"Hausdorff metric requires both v of completion and v of gt")
        dist1 = directed_hausdorff(v, v_gt)[0]
        dist2 = directed_hausdorff(v_gt, v)[0]
        res = max(dist1, dist2)  
    elif (metric == "L2"):
        res = np.linalg.norm(v-v_gt)
    return res

def create_best_animation_by_metric(subject,pose, metric):
    from os import listdir
    from os.path import isfile, join
    dfaust_path = r"C:\Users\ido.iGIP1\hy\Ramp\shape_completion-main\src\core\results\debug_experiment\version_19\completions\DFaustProj"
    onlyfiles = [f for f in listdir(dfaust_path) if isfile(join(dfaust_path, f)) and f.startswith(f"gt_{subject}") and f.endswith("res.ply")]
    files_and_digits = [(f,[int(s) for s in f.split("_") if s.isdigit()][-1]) for f in onlyfiles]
    poses = {}
    for (f,digit) in files_and_digits:
        if digit not in poses:
            poses[digit]=[]
        poses[digit]+=[f]
    file_dir = r"R:\Mano\data\DFaust\DFaust\full\{}\{}".format(subject,pose)
    file_count = len([name for name in os.listdir(file_dir)])
    file_names = [(r"R:\Mano\data\DFaust\DFaust\full\{}\{}".format(subject,pose,d)+"\\{0:05d}.OFF".format(d),d) for d in range(file_count)]
    geometries_gt = [load_off(f) for f,_ in file_names]
    metrics_gt = None
    if(metric != "hausdorff" and metric != "L2"):
        metrics_gt = [calculate_metric(v,f, metric) for v,f in geometries_gt]
    best_reconstruction_by_metric = {}
    poses_sorted = sorted(poses.items())
    for (id,file_paths) in tqdm(poses_sorted):
        geometries_comp = [load_ply(path) for path in file_paths]
        metrics = np.array([calculate_metric(v,f,metric,*geometries_gt[id]) for v,f in geometries_comp])
        current_gt_metric =  metrics_gt[id] if metrics_gt is not None else None
        idx = find_minima(metrics,current_gt_metric,metric)
        #print (id)
        best_reconstruction_by_metric[id]=file_paths[int(idx)]
    anim_plys = best_reconstruction_by_metric.items()
    print(anim_plys)
    print(best_reconstruction_by_metric.keys())
    anim_plys = list(best_reconstruction_by_metric.values())
    # plydata = PlyData.read(dfaust_path)
    plys = [PlyData.read(dfaust_path + "\\" +f) for f in anim_plys]
    faces = np.array(list([list(face) for face in plys[0]["face"].data])).squeeze() 
    # np.array(list([list(vertex) for vertex in vertices]))
    vs = [np.array(list([list(vertex) for vertex in ply["vertex"].data])) for ply in plys]
    return vs,faces, geometries_gt

def animate_subject_pose_by_metric(subject,pose,metric):
    vs,f,_ = create_best_animation_by_metric(subject,pose,metric)
    animate(vs,f,gif_name=r"C:\Users\ido.iGIP1\hy\animate_{}_{}_{}.gif".format(metric, subject, pose))

def multianimate_comparison(subject,pose,metric):
    completion_vs,_, gt_geomtries = create_best_animation_by_metric(subject,pose,metric)
    # file_dir = r"R:\Mano\data\DFaust\DFaust\full\{}\{}".format(subject,pose)
    # file_count = len([name for name in os.listdir(file_dir)])
    # file_names = [(r"R:\Mano\data\DFaust\DFaust\full\{}\{}".format(subject,pose,d)+"\\{0:05d}.OFF".format(d),d) for d in range(file_count)]
    # gt_geomtries = [load_off(f) for f,_ in file_names]
    gt_vs = [vs for (vs,fs) in gt_geomtries]
    f = gt_geomtries[0][1]
    multianimate([completion_vs,gt_vs],[f]*2,gif_name=None,titles=["completion","gt"])

    # print(poses)
#multianimate_comparison(50020,"shake_arms", "volume")
if __name__ == "__main__":
    import meshplex
    import numpy as np

   

    # file = meshio.read(r"R:\Mano\data\DFaust\DFaust\full\50002\hips\00000.OFF","off")
    # _,faces = file.points,file.get_cells_type("triangle")

    from os import listdir
    from os.path import isfile, join
    # onlyfiles = [f for f in listdir(dfaust_path) if isfile(join(dfaust_path, f)) and f.startswith("gt_50007") and f.endswith("res.ply")]
    # files_and_digits = [(f,[int(s) for s in f.split("_") if s.isdigit()][-1]) for f in onlyfiles]
    # animation = {}
    # for (f,digit) in files_and_digits:
    #     if digit not in animation:
    #         animation[digit] = f
    # anim_plys = list(animation.values())
    # # print(dfaust_path+"\\"+anim_plys[0])
    # # print(onlyfiles[4])
    # # print((onlyfiles[4][-12:-8]))
    # # meshio.read(r"R:\Mano\data\DFaust\DFaust\full\50002\hips\00000.OFF","off")
    # # print(onlyfiles[0])
    # plydata = PlyData.read(dfaust_path)
    # plys = [PlyData.read(dfaust_path + "\\" +f) for f in anim_plys]
    # # print(plys[0]["face"])
    # faces = np.array(list([list(face) for face in plys[0]["face"].data])).squeeze() 
    # # np.array(list([list(vertex) for vertex in vertices]))
    # vs = [np.array(list([list(vertex) for vertex in ply["vertex"].data])) for ply in plys][0]

    # mesh = meshplex.MeshTri(np.array(vs), np.array(faces))

    # V = np.sum(mesh.cell_volumes)
    # print(f"volume is {V}")
    # import meshio
    # file = meshio.read(r"R:\Mano\data\DFaust\DFaust\full\50007\jiggle_on_toes\00000.OFF", "off")
    # # print(mesh)
    # # print(file.__dict__)
    # v, f = file.points, file.get_cells_type("triangle")
    # mesh = meshplex.MeshTri(np.array(v), np.array(f))
    # V = np.sum(mesh.cell_volumes)
    # print(f"volume is {V}")
    
    # from plyfile import PlyData, PlyElement
    # file = PlyData.read(r"C:\Users\ido.iGIP1\hy\Ramp\shape_completion-main\src\core\results\debug_experiment\version_19\completions\DFaustProj\gt_50007_jiggle_on_toes_5_2_tp_50007_jiggle_on_toes_0_res.ply")
    # # print(mesh)
    # # print(file.__dict__)
    # v, f = file["vertex"].data, file["face"].data
    # v = np.array([list(vertex) for vertex in v])
    # f = np.array(list([list(face) for face in f])).squeeze() 
    # print(v)
    # mesh = meshplex.MeshTri(np.array(v), f)
    # V2 = np.sum(mesh.cell_volumes)
    # print(f"volume is {V2}")
    # print(f"diff between 2 is {abs(V2-V)}")
    # # animate(vs,faces,gif_name=r"C:\Users\ido.iGIP1\hy\Ramp\animate.gif")
def calc_volume_comp(gt_path,res_path):
    file = meshio.read(gt_path, "off")
    # print(mesh)
    # print(file.__dict__)
    v, f = file.points, file.get_cells_type("triangle")
    mesh = meshplex.MeshTri(np.array(v), np.array(f))
    V = np.sum(mesh.cell_volumes)
    print(f"volume is {V}")
    
    file = PlyData.read(res_path)
    # print(mesh)
    # print(file.__dict__)
    v, f = file["vertex"].data, file["face"].data
    v = np.array([list(vertex) for vertex in v])
    f = np.array(list([list(face) for face in f])).squeeze() 
    print(v)
    mesh = meshplex.MeshTri(np.array(v), f)
    V2 = np.sum(mesh.cell_volumes)
    print(f"volume is {V2}")
    print(f"diff between 2 is {abs(V2-V)}")

def calc_geodesic_comp(gt_path,res_path):
    import meshio
    file = meshio.read(gt_path, "off")
    # print(mesh)
    # print(file.__dict__)
    v, f = file.points, file.get_cells_type("triangle")
    geoalg=geodesic.PyGeodesicAlgorithmExact(v,f)
    print(geoalg)
    source_indices = np.array([0])
    target_indices = None
    distances, best_source = geoalg.geodesicDistances(source_indices, target_indices)
    print(np.mean(distances))

    file = PlyData.read(res_path)
    # print(mesh)
    # print(file.__dict__)
    v, f = file["vertex"].data, file["face"].data
    v = np.array([list(vertex) for vertex in v])
    f = np.array(list([list(face) for face in f])).squeeze() 
    geoalg2=geodesic.PyGeodesicAlgorithmExact(v,f)
    source_indices = np.array([0])
    target_indices = None
    distances, best_source = geoalg2.geodesicDistances(source_indices, target_indices)
    print(np.mean(distances))

    # mesh = meshplex.MeshTri(np.array(v), np.array(f))
    # V = np.sum(mesh.cell_volumes)
    # print(f"volume is {V}")
def calc_hausdorff_dist(gt_path, res_path):
    from scipy.spatial.distance import directed_hausdorff
    file = meshio.read(gt_path, "off")
    v = file.points
    v = np.array([list(vertex) for vertex in v])
    file2 = PlyData.read(res_path)
    v2 = file2["vertex"].data
    v2 = np.array([list(vertex) for vertex in v2])
    dist1 = directed_hausdorff(v, v2)[0]
    dist2 = directed_hausdorff(v2, v)[0]
    symmetric_dist = max(dist1, dist2)   
    print(symmetric_dist)

# calc_hausdorff_dist(r"R:\Mano\data\DFaust\DFaust\full\50007\jiggle_on_toes\00076.OFF",r"C:\Users\ido.iGIP1\hy\Ramp\shape_completion-main\src\core\results\debug_experiment\version_19\completions\DFaustProj\gt_50007_jiggle_on_toes_5_2_tp_50007_jiggle_on_toes_0_res.ply")