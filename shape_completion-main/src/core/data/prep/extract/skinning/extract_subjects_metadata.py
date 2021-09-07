from geom.mesh.io.collad import ColladaFile
import numpy as np
import os
import pathlib
import pickle
# In this script we will extract the meatadata per subject and save in another folder,
# the metadata is the weights, joints Tree Hierarchy, inv bind matrices, rest pose vertices/faces
# TODO - Correct Filepaths
# animation_name = 'Baseball Idle.dae'
# animation_dir = r'\\gip-main\data\ShapeCompletion\Mixamo\spares\collaterals\collada_sequences\MPI-FAUST'
subject_metadata_dir = pathlib.Path(r'R:\MixamoSkinned\index\subject_joint_meta')

import geom.mesh.vis.base as mplt

# subjects = [str(i).zfill(3) for i in range(0, 100, 10)]

# for sub in subjects:
#     animation_path = os.path.join(animation_dir, sub, animation_name)
#     animation = ColladaFile(animation_path)
#     animation.save_subject_metadata(subject_metadata_dir, sub)



def save_reordered_weights_without_joints_names(sub, joints_orders):
    file_path = os.path.join(subject_metadata_dir, sub + '.pkl')
    data = pickle.load(open(file_path, 'rb'))
    weights = []
    for joint in joints_orders:
        weights += [data['weights'][joint]]
    weights = np.stack(weights, axis=0)

    data['reordered_weights'] = weights

    pickle.dump(data, open(file_path, 'wb'))


def save_joints_meta_data():
    data = pickle.load(open(os.path.join(subject_metadata_dir, r"000.pkl"), 'rb'))
    edges = data['joints_tree_hierarchy']
    joints_orders = ['mixamorig_Hips']
    joints_parents = ['']
    for edge in edges:
        joints_parents += [edge[0]]
        joints_orders += [edge[1]]
    parents_by_index = []
    for parent in joints_parents[1:]:
        for index, joint in enumerate(joints_orders):
            if joint == parent:
                parents_by_index += [index]
    print(parents_by_index)
    pickle.dump(
        {'joints_parents': joints_parents, 'joints_orders': joints_orders, 'parents_by_index': parents_by_index},
        open(os.path.join(subject_metadata_dir, r"joints_metadata.pkl"), "wb"))

def name2id(joint_name, joints_orders):
    for index, joint in enumerate(joints_orders):
        if joint == joint_name:
            return index
    return -1

def get_joints_locations(joints_global_trans):
    return np.array([trans[0:3, 3] for trans in joints_global_trans])

if __name__ == '__main__':
    # save_joints_meta_data()
    path_file = os.path.join(subject_metadata_dir, r"joints_metadata.pkl")
    # joints_orders = pickle.load(open(path_file, 'rb'))['joints_orders']
    # subjects = [str(i).zfill(3) for i in range(0, 100, 10)]
    # for sub in subjects:
    #     save_reordered_weights_without_joints_names(sub, joints_orders)

    #adding edge
    joints_metadata = pickle.load(open(path_file, 'rb'))
    joints_orders = joints_metadata['joints_orders']
    data = pickle.load(open(os.path.join(subject_metadata_dir, r"000.pkl"), 'rb'))
    edges = []
    for edge in data['joints_tree_hierarchy']:
        edges += [[name2id(edge[0], joints_orders), name2id(edge[1], joints_orders)]]
    edges = np.array(edges)
    joints_metadata['edges'] = edges
    pickle.dump(joints_metadata, open(path_file, "wb"))

    joint_transformation = np.array([np.linalg.inv(data['inv_bind_matrices'][joint]) for joint in joints_orders])
    p = mplt.plotter()
    mplt.add_mesh(p, v=data['rest_vertices'], f=data['faces'], grid_on=True, strategy='mesh', opacity=0.3, clr='coral')
    mplt.add_skeleton(p=p, v=get_joints_locations(joint_transformation), edges=edges, transformations=joint_transformation, scale=0.1)
    p.show()
