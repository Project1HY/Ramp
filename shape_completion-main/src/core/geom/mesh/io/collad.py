import collada
import numpy as np
import pyvista as pv
import geom.mesh.vis.base as mplt
import os
import pickle


# A helper class for joint's representation
class ColladaJoint:
    # TODO
    # must put constrain on the type of parent, must be Joint
    def __init__(self, name, id, inv_trans_mat, children, keyframes_transf, parent_name):
        self.name = name
        self.id = id
        self.inv_trans_mat = inv_trans_mat
        self.children = children
        self.keyframes_transf = keyframes_transf
        self.parent_name = parent_name

    def get_parent_name(self):
        return self.parent_name

    def get_name(self):
        return self.name


# collada loader class
class ColladaFile:
    def __init__(self, file_name, is_converted_file=True):

        self.file_path = file_name

        self.file_name = os.path.splitext(os.path.split(file_name)[-1])[0]

        self.is_converted_file = is_converted_file

        self.model = collada.Collada(self.file_path)

        # joint inverse bind-pose matrices
        # key: joint_name --> value: inverse transformation matrix
        self.inv_joints_trans = self.model.controllers[0].joint_matrices

        # bind-pose matrix - Provides extra information about the position and
        # orientation of the base mesh before binding.
        # TODO mul it with the vertices
        self.bind_pose_matrix = self.model.controllers[0].bind_shape_matrix

        # we assign foreach joint a number
        # key: joint_name --> value: a unique number
        self.joint_name2id = {joint_name: index for index, joint_name in
                              enumerate(self.inv_joints_trans.keys())}

        self.root_joint = self.get_root_joint()

        self.joints = self.get_joints_as_dict()

        self.rest_pose_global_transf = self.get_rest_pose_transformation(self.model.scene.nodes[1])

        self.keyframes = self.load_keyframes()

        self.weights = self.get_weights()

        self.vertices = self.get_vertices()

        self.faces = self.get_triangles()

        # dict joint_id --> joint_name
        self.joint_id2name = self.get_joints_id_name_map()

    # TODO uses of this method
    def get_joint_id(self, joint_node):
        return self.joint_name2id[joint_node.id]

    def get_joint_children(self, joint_node):
        children = []
        for index, child in enumerate(joint_node.children):

            if self.is_converted_file and index == 0:
                continue

            if child.id in self.joint_name2id.keys():
                name = child.id
                id = self.joint_name2id[child.id]
                inv_trans = child.matrix
                childern = self.get_joint_children(child)
                # keyframes = self.keyframes[name]
                parent_name = joint_node.id

                # joint = Joint(name, id, inv_trans, childern, keyframes, parent_name)
                joint = ColladaJoint(name, id, inv_trans, childern, None, parent_name)

                children += [joint]
        return children

    def get_root_joint(self):
        # the node of the root joint
        root_joint_node = self.model.scene.nodes[1]

        name = root_joint_node.id
        id = self.get_joint_id(root_joint_node)
        childern = self.get_joint_children(root_joint_node)
        inv_trans = root_joint_node.matrix
        # keyframes = self.keyframes[name]
        keyframes = None
        parent_name = None

        return ColladaJoint(name, id, inv_trans, childern, keyframes, parent_name)

    def get_rest_pose_transformation(self, joint_node):
        stack = []
        global_transformation = {}

        global_transformation[joint_node.id] = joint_node.matrix

        stack += joint_node.children
        while (len(stack) != 0):
            node = stack.pop()
            if len(node.xmlnode) > 1:
                node_name = node.id
                if node_name in self.joints.keys():
                    global_transformation[node_name] = node.matrix
                    stack += node.children

        return global_transformation

    def get_joint_matrices(self, keyframe_num):
        stack = []
        joint_matrices = {}
        joint_matrices[self.root_joint.name] = self.keyframes[self.root_joint.get_name()][keyframe_num, :, :]
        stack += self.root_joint.children
        while (len(stack) != 0):
            joint = stack.pop()
            joint_parent_name = joint.get_parent_name()
            joint_transformation = self.keyframes[joint.get_name()][keyframe_num, :, :]
            joint_world_matrix = np.matmul(joint_matrices[joint_parent_name], joint_transformation)
            joint_matrices[joint.name] = joint_world_matrix
            stack += joint.children
        return joint_matrices

    def get_keyframe_skeleton(self, keyframe_index):

        joint_matrices = self.get_joint_matrices(keyframe_index)
        # reordering the transformation
        reordered_transformation = [None] * len(joint_matrices)
        for joint_name, joint_transf in joint_matrices.items():
            reordered_transformation[self.joint_name2id[joint_name]] = joint_transf

        v = np.array([transf[0:3, 3] for transf in reordered_transformation])
        edges = np.array(self.get_edges(self.model.scene.nodes[1]))
        return v, edges, reordered_transformation

    def joint_matrices_reordering(self, joint_matrices):
        # converting from a dictionary of transformations to a list of transformation
        reordered_transformation = [None] * len(joint_matrices)
        for joint_name, joint_transf in joint_matrices.items():
            reordered_transformation[self.joint_name2id[joint_name]] = joint_transf
        return reordered_transformation

    def get_edges(self, node):
        edges = []
        for child in node.children:
            if type(child) == collada.scene.Node:
                if node.id in self.joint_name2id.keys() and child.id in self.joint_name2id.keys():
                    edges += [[self.joint_name2id[node.id], self.joint_name2id[child.id]]]
                child_edges = self.get_edges(child)
                edges += child_edges
        return edges

    def load_keyframes(self):
        suffix = '-Matrix-animation-output-transform'
        animation = {
            joint_animation.name: joint_animation.sourceById[joint_animation.name + suffix].data.reshape(-1, 4, 4) for
            joint_animation in self.model.animations}
        if not self.is_converted_file:
            return animation
        else:
            max_keyframes_num = None
            min_keyframes_num = None
            idle_joints = []
            for joint_name, joint_transformation in animation.items():
                if len(joint_transformation) == 0:
                    idle_joints += [joint_name]
                    continue

                if max_keyframes_num is None and min_keyframes_num is None:
                    max_keyframes_num = len(joint_transformation)
                    min_keyframes_num = len(joint_transformation)
                elif max_keyframes_num < len(joint_transformation):
                    max_keyframes_num = len(joint_transformation)
                elif min_keyframes_num > len(joint_transformation):
                    min_keyframes_num = len(joint_transformation)

            problematic_joints = {}
            for joint_name, joint_transformation in animation.items():
                if len(joint_transformation) == 0:
                    rest_pose_trans = self.rest_pose_global_transf[joint_name]
                    animation[joint_name] = np.tile(rest_pose_trans, (min_keyframes_num, 1, 1))
                else:
                    if len(joint_transformation) < max_keyframes_num:
                        problematic_joints[joint_name] = len(joint_transformation)
                    animation[joint_name] = joint_transformation[0:min_keyframes_num, :, :]

            self.max_keyframes_num = max_keyframes_num
            self.min_keyframes_num = min_keyframes_num
            self.problematic_joints = problematic_joints
            self.idle_joints = idle_joints
            return animation

    def get_vertices(self):
        return self.model.geometries[0].primitives[0].vertex

    def get_triangles(self):
        return self.model.geometries[0].primitives[0].vertex_index.reshape((-1, 3))

    def get_joints_as_dict(self):
        stack = [self.root_joint]
        joint_dict = {}
        while (len(stack) != 0):
            joint = stack.pop()
            joint_dict[joint.get_name()] = joint
            stack += joint.children
        return joint_dict

    def get_weights(self):
        # the same as self.joint_ids
        weight_joints = self.model.controllers[0].weight_joints.data
        # v
        vertex_weight_index = self.model.controllers[0].vertex_weight_index
        # weights
        weights = self.model.controllers[0].weights
        # vcount
        vcounts = self.model.controllers[0].vcounts

        our_weights = np.zeros((weight_joints.size, vcounts.size))

        counter = 0
        for vertex_index, vertex_bones_num in enumerate(vcounts):
            for i in range(vertex_bones_num):
                joint_index = self.joint_name2id[weight_joints[vertex_weight_index[counter]][0]]
                vertex_weight = weights[vertex_weight_index[counter + 1]]

                counter += 2

                our_weights[joint_index, vertex_index] = vertex_weight

        return our_weights

    def get_weights_as_dict_with_joint_names(self):
        dict = {}
        for joint_index, joint_weights in enumerate(self.weights):
            joint_name = self.joint_id2name[str(joint_index)]
            dict[joint_name] = joint_weights
        return dict

    def vertices_per_keyframe(self, keyframe):
        # get joint matrices and doing reordering
        joint_matrices = self.joint_matrices_reordering(self.get_joint_matrices(keyframe))
        inv_joints_trans = self.joint_matrices_reordering(self.inv_joints_trans)
        skinned_vertices = np.zeros_like(self.vertices)

        for vertex_index, vertex in enumerate(self.vertices):
            h_vertex = np.ones((4, 1))
            h_vertex[0:3, :] = vertex.reshape(3, 1)
            skinned_vertex = np.zeros_like(h_vertex)

            for joint_index, inv_joint_vertex in enumerate(inv_joints_trans):
                weight = self.weights[joint_index, vertex_index]
                joint_matrix = joint_matrices[joint_index]
                skinned_vertex += weight * (np.matmul(joint_matrix, np.matmul(inv_joint_vertex, h_vertex)))

            skinned_vertices[vertex_index] = skinned_vertex.reshape(4)[0:3]

        return skinned_vertices

    def get_t_pose_vertices_from_keyframe_vertices(self, keyframe):
        skinned_vertices = self.vertices_per_keyframe(keyframe)

        # get joint matrices and doing reordering
        joint_matrices = self.joint_matrices_reordering(self.get_joint_matrices(keyframe))
        inv_joints_trans = self.joint_matrices_reordering(self.inv_joints_trans)
        t_pose_vertices = np.zeros_like(skinned_vertices)

        for vertex_index, vertex in enumerate(skinned_vertices):
            h_vertex = np.ones((4, 1))
            h_vertex[0:3, :] = vertex.reshape(3, 1)
            t_pose_vertex = np.zeros_like(h_vertex)

            for joint_index, inv_joint_vertex in enumerate(inv_joints_trans):
                weight = self.weights[joint_index, vertex_index]
                joint_matrix = joint_matrices[joint_index]
                t_pose_vertex += weight * (np.matmul(np.linalg.inv(np.matmul(joint_matrix, inv_joint_vertex)), h_vertex))

            t_pose_vertices[vertex_index] = t_pose_vertex.reshape(4)[0:3]

        return t_pose_vertices

    def get_joints_trans_i2j(self, keyframe_i, keyframe_j):
        joint_matrices_i = self.joint_matrices_reordering(self.get_joint_matrices(keyframe_i))
        joint_matrices_j = self.joint_matrices_reordering(self.get_joint_matrices(keyframe_j))
        inv_joints_trans = self.joint_matrices_reordering(self.inv_joints_trans)

        joints_trans_i2j = [None]*len(joint_matrices_i)
        for joint_index, inv_joint_vertex in enumerate(inv_joints_trans):
            joint_matrix_i = joint_matrices_i[joint_index]
            joint_matrix_j = joint_matrices_j[joint_index]

            joints_trans_i2j[joint_index] = np.matmul(np.matmul(joint_matrix_j, inv_joint_vertex), np.linalg.inv(np.matmul(joint_matrix_i, inv_joint_vertex)))

        return joints_trans_i2j


    def skin_i2j(self, keyframe_i, keyframe_j):
        vertices_i = self.vertices_per_keyframe(keyframe_i)
        joints_trans_i2j = self.get_joints_trans_i2j(keyframe_i, keyframe_j)
        skinned_vertices = np.zeros_like(vertices_i)

        for vertex_index, vertex in enumerate(vertices_i):
            h_vertex = np.ones((4, 1))
            h_vertex[0:3, :] = vertex.reshape(3, 1)
            skinned_vertex = np.zeros_like(h_vertex)

            for joint_index, joint_trans in enumerate(joints_trans_i2j):
                weight = self.weights[joint_index, vertex_index]
                skinned_vertex += weight * (np.matmul(joint_trans, h_vertex))

            skinned_vertices[vertex_index] = skinned_vertex.reshape(4)[0:3]

        return skinned_vertices

    def plot_rest_pose(self):
        reordered_inv_bind_pose_trans = self.joint_matrices_reordering(self.inv_joints_trans)
        reordered_bind_pose_trans = [np.linalg.inv(inv_bind_mat) for inv_bind_mat in reordered_inv_bind_pose_trans]
        v_skeleton = np.array([transf[0:3, 3] for transf in reordered_bind_pose_trans])
        edges = np.array(self.get_edges(self.model.scene.nodes[1]))

        p = mplt.plotter()
        mplt.add_mesh(p, v=self.vertices, f=self.faces, grid_on=True, strategy='mesh', opacity=0.3, clr='coral')
        mplt.add_skeleton(p=p, v=v_skeleton, edges=edges, transformations=reordered_bind_pose_trans, scale=0.1)
        p.show()

    def plot_keyframe_mesh(self, keyframe_index):
        v_skeleton, edges, joint_transformation = self.get_keyframe_skeleton(keyframe_index)

        vertices_skinned = self.vertices_per_keyframe(keyframe_index)

        p = mplt.plotter()
        mplt.add_mesh(p, v=vertices_skinned, f=self.faces, grid_on=True, strategy='mesh', opacity=0.3, clr='coral')
        mplt.add_skeleton(p=p, v=v_skeleton, edges=edges, transformations=joint_transformation, scale=0.1)
        p.show()

    def weight_montage(self, keyframe_index, num_weights_per_screen=12):
        """
        Draws all joints in a montage on the mesh defined by the supplied keyframe
        :param int keyframe_index: The keyframe id
        """
        # Get vertices:
        v = self.vertices_per_keyframe(keyframe_index)

        # Insert joint coordinate system per mesh support:
        _, _, joint_trans_mats = self.get_keyframe_skeleton(keyframe_index)
        from functools import partial
        from geom.mesh.vis.base import add_joint_coordinate_system

        def append_transformation_to_weight(transformations, plotter, _, mesh_index):
            add_joint_coordinate_system(plotter, trans_mats=transformations[mesh_index], scale=0.5)

        # Split the weights per num_weights_per_screen
        weight_screen_split = np.arange(0, len(self.weights), num_weights_per_screen)[1:]
        for (vb, wb, tb) in zip(np.split([v] * len(self.weights), weight_screen_split),
                                np.split(self.weights, weight_screen_split),
                                np.split(joint_trans_mats, weight_screen_split)):
            ext_func = partial(append_transformation_to_weight, tb)
            mplt.plot_mesh_montage(vb=vb, fb=self.faces, clrb=wb, strategy='mesh', ext_func=ext_func, opacity=1,
                                   lighting=True)

    def write_keyframe_as_obj(self, keyframe_index, file_path=str()):
        vertices = self.vertices_per_keyframe(keyframe_index)

        with open(file_path, 'w') as f:
            f.write("# OBJ file\n")
            for v in vertices:
                f.write("v %.4f %.4f %.4f\n" % (v[0], v[1], v[2]))
            for p in self.faces:
                f.write("f")
                for i in p:
                    f.write(" %d" % (i + 1))
                f.write("\n")

    def save_weights(self, file_name):
        np.save(file_name, self.weights)

    def save_inv_bind_matrices(self, file_name):
        a = np.stack(self.inv_joints_trans.values())
        np.save(file_name, a)

    def save_vertices(self, file_name):
        np.save(file_name, self.vertices)

    def save_faces(self, file_name):
        np.save(file_name, self.faces)

    def save_lines(self, file_name):
        lines = self.get_edges(self.model.scene.nodes[1])
        np.save(file_name, lines)

    def save_subject_metadata(self, directory_path, subject):
        dict = {}
        dict['inv_bind_matrices'] = self.inv_joints_trans
        dict['joint_name2id'] = self.joint_name2id
        dict['joint_id2name'] = self.joint_id2name
        dict['rest_vertices'] = self.vertices
        dict['faces'] = self.faces
        dict['weights'] = self.get_weights_as_dict_with_joint_names()

        tree = []
        for edge in self.get_edges(self.model.scene.nodes[1]):
            u, v = edge[0], edge[1]
            tree += [[self.joint_id2name[str(u)], self.joint_id2name[str(v)]]]
        dict['joints_tree_hierarchy'] = tree

        path = os.path.join(directory_path, subject + '.pkl')

        self.write_pickle_file(file_path=path, dictionary=dict)

    def save_animation_keyframes_as_obj_files(self, directory_path):
        animation_transformation = np.stack(self.keyframes.values())
        for i in range(animation_transformation.shape[1]):
            obj_name = str(i + 1).zfill(3) + '.obj'
            path = os.path.join(directory_path, obj_name)
            self.write_keyframe_as_obj(i, path)

    def save_keyframe_metadata(self, keyframe_index, directory_path):
        frame_meta_data_name = str(keyframe_index + 1).zfill(3) + '.pkl'
        path = os.path.join(directory_path, frame_meta_data_name)

        local_transformation = {joint_name: local_transf[keyframe_index, :, :] for joint_name, local_transf in
                                self.keyframes.items()}
        global_transformation = self.get_joint_matrices(keyframe_index)

        transformation = {'local': local_transformation, 'global': global_transformation}

        self.write_pickle_file(file_path=path, dictionary=transformation)

    def write_to_log(self):
        with open(self.log_path, "a") as logger:
            lines = [self.file_path, f'max keyframes: {self.max_keyframes_num}',
                     f'min keyframes: {self.min_keyframes_num}']
            for line in lines:
                logger.writelines(line + '\n')

    def save_animation_meta_data(self, directory_path):
        animation_transformation = np.stack(self.keyframes.values())
        for i in range(animation_transformation.shape[1]):
            self.save_keyframe_metadata(i, directory_path)

    def get_frame_num(self):
        return np.stack(self.keyframes.values()).shape[1]

    def get_joints_id_name_map(self):
        dict = {}
        for joint_name, joint_id in self.joint_name2id.items():
            dict[str(joint_id)] = joint_name
        return dict

    @staticmethod
    def write_pickle_file(file_path, dictionary):
        with open(file_path, 'wb') as f:
            pickle.dump(dictionary, f, pickle.HIGHEST_PROTOCOL)

    def save_statiscis(self, directory_path):
        dict = {}
        dict['max_keyframes_num'] = self.max_keyframes_num
        dict['problematic_joints'] = self.problematic_joints
        dict['idle_joints'] = self.idle_joints

        file_name = self.file_name + '.pkl'
        path = os.path.join(directory_path, file_name)

        self.write_pickle_file(file_path=path, dictionary=dict)


if __name__ == '__main__':
    # file_name = os.path.join('\\\\gip-main', 'data', 'ShapeCompletion', 'Mixamo', 'spares', 'collaterals',
    #                          'collada_sequences', 'MPI-FAUST', '000', 'Crouch Running.dae')

    file_name = r'Big Jump.dae'
    animation = ColladaFile(file_name)
    # animation.plot_rest_pose()
    animation.plot_keyframe_mesh(20)
    # p = mplt.plotter()
    # vertices = animation.skin_i2j(10, 20)
    # mplt.add_mesh(p, v=vertices, f=animation.faces, grid_on=True, strategy='mesh', opacity=0.3, clr='coral')
    # p.show()

    # # animation.plot_rest_pose()
    # # animation.plot_keyframe_mesh(keyframe_index=40)
    # # animation.weight_montage(keyframe_index=40)
    #
    # vertices_skinned = animation.vertices_per_keyframe(4)
    #
    # p = mplt.plotter()
    # mplt.add_mesh(p, v=vertices_skinned, f=animation.faces, grid_on=True, strategy='mesh', opacity=0.3, clr='red')
    # # p.show()
    # import torch
    # v = torch.tensor(vertices_skinned).unsqueeze(0)
    # skinning_weights = torch.tensor(animation.weights).unsqueeze(0)
    # v_homo = torch.cat((v, torch.ones(v.shape[0], v.shape[1], 1)), 2)
    # T = torch.eye(4).repeat(1, 52, 1, 1)
    # v_new = torch.einsum('bjki,bni, bjn->bjnk', T[:, :, :3, :4], v_homo, skinning_weights)  # [B x J x N x 3]
    # v_new = torch.sum(v_new, dim=1)  # [B x N x 3]
    # v_new = v_new.squeeze(0).cpu().detach().numpy()
    # (v_new == vertices_skinned).all()
    #
    # # p = mplt.plotter()
    # mplt.add_mesh(p, v=v_new, f=animation.faces, grid_on=True, strategy='mesh', opacity=0.3, clr='blue')
    # p.show()
    # animation.save_subject_metadata('tmp', '000')
    # animation.write_statiscis('tmp')
    # with open(r'R:\MixamoSkinned\statistics\010\Standing Melee Combo Attack Ver. 1.pkl', 'rb') as f:
    #     dict = pickle.load(f)
    #     dict['local']
