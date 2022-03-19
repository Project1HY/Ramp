import os
import torch
#import tqdm
import PySimpleGUI as sg
import numpy as np
import copy
import pyvista as pv
from vtkmodules.vtkRenderingOpenGL2 import vtkOpenGLActor
from scipy.spatial import KDTree
from human_body_prior.body_model.body_model import BodyModel
from save_load_obj import save_obj,load_obj
from human_mesh_utils import get_body_mesh_of_single_frame
import human_mesh_utils
import trimesh

#hardcoded for now:
num_betas =16
num_dmpls =8

class AmassActor:
    def __init__(self,initial_model_npz:str,smplh_dir:str, dmpl_dir:str,num_betas :int,num_dmpls :int,comp_device:str):
        self.comp_device=comp_device
        self.max_frame:int=0
        self.cur_frame:int=0
        self.num_betas:int=num_betas
        self.num_dmpls:int=num_dmpls
        self.male_body_model:BodyModel=human_mesh_utils.get_body_model(gender='male',smplh_dir=smplh_dir, dmpl_dir=dmpl_dir,num_betas=num_betas,num_dmpls=num_dmpls,comp_device=self.comp_device)
        self.female_body_model:BodyModel=human_mesh_utils.get_body_model(gender='female',smplh_dir=smplh_dir, dmpl_dir=dmpl_dir,num_betas=num_betas,num_dmpls=num_dmpls,comp_device=self.comp_device)
        self.neutral_body_model:BodyModel=human_mesh_utils.get_body_model(gender='neutral',smplh_dir=smplh_dir, dmpl_dir=dmpl_dir,num_betas=num_betas,num_dmpls=num_dmpls,comp_device=self.comp_device)
        self.cur_gender='neutral'
        # TODO make sure after the init all the following varibles are not none
        self._cur_model_npz_file:str=initial_model_npz
        self._cur_body_pose:dict=None
        self._cur_body_shape:dict=None
        #cur_body_params is a merge between _cur_body_pose and _cur_body_shape
        self.cur_body_params:dict=None
        self.cur_body_model:BodyModel=None
        # init
        self.load_new_model_from_model_npz_file(model_npz=self._cur_model_npz_file)

    def _update_frame(self,new_frame:int):
        self.cur_frame=max(min(new_frame,self.max_frame),0)

    def _is_parameter_initialized(self,param:np.array)->bool:
        return param.size(1)!=0

    def _is_pose_initialized(self)->bool:
        if self._cur_body_pose==None:
            return False
        for paramName in human_mesh_utils.get_pose_params_list():
            if not self._is_parameter_initialized(self._cur_body_pose[paramName]):
                return False
        return True

    def _is_shape_initialized(self)->bool:
        if self._cur_body_shape==None:
            return False
        for paramName in human_mesh_utils.get_shape_params_list():
            if not self._is_parameter_initialized(self._cur_body_shape[paramName]):
                return False
        return True

    def _is_pose_and_shape_initialized(self)->bool:
        return self._is_pose_initialized() and self._is_shape_initialized()

    def _update_body_params(self):
        if self._is_pose_and_shape_initialized():
            self.cur_body_params=human_mesh_utils.get_body_params_from_pose_and_shape_params(shape_body_params=self._cur_body_shape,pose_body_params=self._cur_body_pose,comp_device=self.comp_device)
            self.cur_body_model=human_mesh_utils.update_body_model_with_body_params(body_model=self._get_cur_body_model(),body_parms=self.cur_body_params)

    def reset_gender(self):
        self.update_gender(human_mesh_utils.get_gender_from_model_file(model_npz=self._cur_model_npz_file))

    def update_gender(self,gender:str):
        if gender in human_mesh_utils.get_gender_list():
            self.cur_gender=gender
        self._update_body_params()

    def update_frame_with_frame_delta(self,frame_delta:int):
        self._update_frame(new_frame=self.cur_frame+frame_delta)

    def update_body_shape_from_model_npz_file(self,model_npz:str):
        if not human_mesh_utils.is_valid_npz_file(model_npz=model_npz):
            print('not valid model_npz file')
            return
        self._cur_body_shape=None
        self._cur_body_shape=human_mesh_utils.get_shape_body_params(model_npz=model_npz,num_betas=self.num_betas,comp_device=self.comp_device)
        self.cur_gender=human_mesh_utils.get_gender_from_model_file(model_npz=model_npz)
        self._update_body_params()

    def update_body_pose_from_model_npz_file(self,model_npz:str):
        if not human_mesh_utils.is_valid_npz_file(model_npz=model_npz):
            print('not valid model_npz file')
            return
        self.cur_frame=0
        self._cur_body_pose=None
        self.max_frame=human_mesh_utils.get_number_of_frames(model_npz=model_npz)-1
        self._cur_body_pose=human_mesh_utils.get_pose_body_params(model_npz=model_npz,num_dmpls=self.num_dmpls,comp_device=self.comp_device)
        self._update_body_params()

    def load_new_model_from_model_npz_file(self,model_npz:str):
        if not human_mesh_utils.is_valid_npz_file(model_npz=model_npz):
            print('not valid model_npz file')
            return
        self._cur_model_npz_file=model_npz
        self.update_body_shape_from_model_npz_file(model_npz=model_npz)
        self.update_body_pose_from_model_npz_file(model_npz=model_npz)
        self._update_body_params()

    def _get_cur_body_model(self)->BodyModel:
        if self.cur_gender == 'male':
            return self.male_body_model
        elif self.cur_gender == 'female':
            return self.female_body_model
        else: # self.cur_gender == 'neutral':
            return self.neutral_body_model

    def get_cur_frame(self):
        return self.cur_frame

    def get_cur_gender(self):
        return self.cur_gender

    def get_cur_body_params(self):
        return self.cur_body_params

    def get_body_mesh(self)->trimesh.Trimesh:
        return human_mesh_utils.get_body_mesh_of_single_frame(body_model=self.cur_body_model,frameID=self.cur_frame)

    def is_on_last_frame(self)->bool:
        return self.cur_frame==self.max_frame

    def is_on_first_frame(self)->bool:
        return self.cur_frame==0

    def reset_body_pose(self):
        self.load_new_model_from_model_npz_file(self._cur_model_npz_file)

    def update_body_param_from_key_and_value(self,key:str,value:str):
        self.cur_body_params=get_updated_body_param_from_key_and_value(body_params=self.cur_body_params,delim=get_param_delim(),key=key,value=value,cur_frame=self.cur_frame,max_frame=self.max_frame)
        self.cur_body_model=human_mesh_utils.update_body_model_with_body_params(body_model=self._get_cur_body_model(),body_parms=self.cur_body_params)

    def get_max_frame_num(self):
        return self.max_frame

class AmassPlotter:
    def __init__(self):
        self.lastActor:vtkOpenGLActor=None
        self.plotter = pv.Plotter()
        self.cur_body_mesh = trimesh.Trimesh
        #self.effected_triangles:bool = False
        #self.track_vertex_list:bool = True #hradcoded for now
        self._coloring_method:str='none' #can be 'none','track_vertex_list' or 'effected_triangles'
        self.vertex_list_to_track:list = list(range(10)) # hardcoded for now 
    def _update_frame(self,body_mesh:trimesh.Trimesh):
        self.old_body_mesh = self.cur_body_mesh
        self.cur_body_mesh = body_mesh
        #body_mesh = pv.wrap(body_mesh)
        if self.lastActor!=None:
            #self.plotter.remove_actor(self.lastActor)
            self.plotter.update_coordinates(pv.wrap(self.cur_body_mesh).points, render=False)
            if self._coloring_method != 'none':
                colors=self._get_colors_for_plot_mesh()
                self.plotter.update_scalars(colors , render=False)
                self.plotter.update_scalar_bar_range([min(colors),max(max(colors),0.001)])
                #self.plotter.update_scalar_bar_range([0,1]) #scalar bar can be fixed with this option
        else:
            self.add_mesh()
        #self.lastActor= self.plotter.add_mesh(body_mesh, show_edges=False)
        self.plotter.update()
    def update_frame_from_amass_actor(self,amass_actor:AmassActor):
        self._update_frame(body_mesh=amass_actor.get_body_mesh())
    def add_mesh(self,show_edges:bool=False)->None:
        colors=self._get_colors_for_plot_mesh()
        self.lastActor= self.plotter.add_mesh(pv.wrap(self.cur_body_mesh), show_edges=show_edges,scalars=colors)

    def show_interactive_mode(self):
        self.plotter.show(interactive_update=True,interactive=True)
    def rotate(self):
        self.plotter.show(auto_close=False)
    """
    def _custom_colors_needed()->bool:
        return self.effected_triangles or self.track_vertex_list
    """
    def _get_colors_for_plot_mesh(self)->np.array:
        if self._coloring_method == 'track_vertex_list':
            colors = np.zeros(len(self.old_body_mesh.vertices))
            colors[self.vertex_list_to_track]=1
        elif self._coloring_method == 'effected_triangles':
            #TODO remove this in the futhre and write the normal way with vertex coorespondance
            #tree = KDTree(self.old_body_mesh.vertices)
            #colors,_=tree.query(self.cur_body_mesh.vertices)
            colors=np.linalg.norm(self.old_body_mesh.vertices-self.cur_body_mesh.vertices,axis=1)
        else:
            colors=None
        return colors

    def set_coloring_method(self,coloring_method:str):
        assert coloring_method in ['none','track_vertex_list','effected_triangles']
        self._coloring_method=coloring_method
        if self.lastActor!=None:
            self.plotter.remove_actor(self.lastActor)
        self.add_mesh()
    def get_coloring_method(self)->str:
        return self._coloring_method

    """
    def set_effected_triangels(self,effected_triangles:bool):
        self.effected_triangles=effected_triangles
        if self.effected_triangles:
            self.track_vertex_list=False
    def set_track_vertex_list(self,track_vertex_list:bool):
        self.track_vertex_list=track_vertex_list
        if self.track_vertex_list:
            self.effected_triangles=False
    """

#FIXME some code duplication with cli_utils..maybe fix it in the futhure
def assertVaildInputGUI(args)->None:
    human_mesh_utils.assertVaildEnviroment()
    errStr = ''
    # make sure skmplh and dmpl paths are ok
    valid_prior_dirs,err_of_render_list = human_mesh_utils.is_valid_skmpl_and_dmpl_dirs(args)#TODO make it look better
    if not valid_prior_dirs:
        errStr+=err_of_render_list
    if not os.path.exists(args.initial_model_npz):
                errStr+='model file {} not exists'.format(args.initial_model_npz)
    if errStr == '' and not human_mesh_utils.is_valid_npz_file(args.initial_model_npz):
                errStr+='model file {} exists, but not valid'.format(args.initial_model_npz)
    if errStr != '':
        raise Exception("unvalid input:\n"+errStr)

def createLayoutForParameter(param_name:str,parm_length:int,num_elements_in_row:int=10)->list:
    if num_elements_in_row %2 !=0:
        num_elements_in_row=num_elements_in_row+1
        print('num_elements_in_row is odd number .using {} insted'.format(num_elements_in_row))
    header =[sg.Text('{}:'.format(param_name), size=(30,1), justification='left', background_color="#272533")]
    data = []
    for i in range(parm_length):
        data.append(sg.Text('{}:'.format(i), justification='left', background_color="#272533"))
        data.append(sg.Input('0',key=get_key_param(param_name,i), size=(7,1)))
        data.append(sg.Button('p',key=get_key_animate(param_name,i), size=(1,1)))
    #FIXME I know it's ugly but it's fast and dirty
    num_elements_in_row=3*num_elements_in_row #I consider each element as number and value tuple.
    data_chunks = [data[x:x+num_elements_in_row] for x in range(0,len(data),num_elements_in_row)]
    return [header,data_chunks]

def get_key(param_name:str,num:int,delim:str)->str:
    key='{}{}{}'.format(param_name,delim,num)
    return key

def get_entries_from_key(key:str,delim:str)->(str,int):
    param_name,num=key.split(delim,1)
    return param_name,int(num)

def get_animate_delim()->str:
    return "&"

def get_param_delim()->str:
    return ":"

def get_key_param(param_name:str,num:int)->str:
    return get_key(param_name=param_name,num=num,delim=get_param_delim())

def get_entries_from_key_param(key:str)->(str,int):
    return get_entries_from_key(key=key,delim=get_param_delim())

def get_key_animate(param_name:str,num:int)->str:
    return get_key(param_name=param_name,num=num,delim=get_animate_delim())

def get_entries_from_key_animate(key:str)->(str,int):
    return get_entries_from_key(key=key,delim=get_animate_delim())

def get_updated_body_param_from_key_and_value(body_params:dict,delim:str,key:str,value:str,cur_frame:int,max_frame:int)->dict:
    errStr =''
    param_name,num=get_entries_from_key(key=key,delim=delim)
    if param_name not in body_params:
        errStr+='warning update_body_param_from_key_and_value did not contain the param_name.'
    if not isfloat(value):
        errStr+='warning update_body_param_from_key_and_value did not get any float'
    if errStr != '':
        print(errStr)
        return
    value = float(value)
    if param_name not in human_mesh_utils.get_shape_params_list():
        #update for single frame
        body_params[param_name][cur_frame][num]=value
    else:
        #update for all frames for shape list
        # note can be optimized ( this for loop should not be activated when we use the "param animation bottom 'p'")
        for frame in range(0,max_frame+1):
            body_params[param_name][frame][num]=value
    return body_params

def get_all_keys_layout()->list:
    return [createLayoutForParameter(param_name,_len) for param_name,_len in get_param_names_and_lengths()]

"""
delete if not neccery
def get_pose_param_names()->list:
    return ['pose_hand', 'pose_body']
"""

def get_pose_param_names_and_lengths()->list:
    all_param_names_and_lengths = get_param_names_and_lengths()
    pose_param_names = human_mesh_utils.get_pose_params_list()
    res = []
    for param_name,_len in all_param_names_and_lengths:
        if param_name in pose_param_names:
            res.append((param_name,_len))
    return res

def get_param_names_and_lengths()->list:
    #hardcoded for now
    #uncommnet for debug
    """
    return [('pose_hand',3),
    ('pose_body',2)]
    """

    return [('pose_hand',2),
    ('betas',2),
    ('pose_body',3)]
    """

    #original
    return [('root_orient',3),
    ('trans',3),
    ('pose_body',63),
    ('pose_hand',90),
    ('betas',num_betas),
    ('dmpls',num_dmpls)]
    """

def update_GUI_param_data(window:sg.Window,body_params:dict,frameID:int):
    #update body param data
    for param_name,_len in get_param_names_and_lengths():
        for i in range(_len):
            window[get_key_param(param_name,i)].update(value=float(body_params[param_name][frameID][i]))
    window.refresh()

def update_GUI_param_frame(window:sg.Window,frameID:int):
    window['_DISPLAY_FRAME_'].update(value='frame:{}'.format(frameID))
    window.refresh()

def update_GUI_param_frame_delta(window:sg.Window,delta:int):
    window['_DISPLAY_FRAME_DELTA_'].update(value='f.delta:{}'.format(delta))
    window.refresh()

def update_GUI_param_gender(window:sg.Window,gender:str):
    window['_DISPLAY_GENDER_'].update(value='gender:{}'.format(gender))
    window.refresh()

def update_GUI(window:sg.Window,amass_actor:AmassActor,delta:int):
    update_GUI_param_data(window=window,body_params=amass_actor.get_cur_body_params(),frameID=amass_actor.get_cur_frame())
    #update frame delta
    update_GUI_param_frame_delta(window,delta)
    update_GUI_param_gender(window,amass_actor.get_cur_gender())
    #update frame num
    update_GUI_param_frame(window,frameID=amass_actor.get_cur_frame())
    window.refresh()

def isfloat(value):
    try:
      float(value)
      return True
    except ValueError:
      return False

def get_GUI()->sg.Window:
    bt: dict = {'size':(7,2), 'font':('Franklin Gothic Book', 8), 'button_color':("black","#F1EABC")}
    #bt_bigger: dict = {'size':(17,2), 'font':('Franklin Gothic Book', 24), 'button_color':("black","#F1EABC")}
    layout: list = [
        [sg.Text('AMASS Mesh Editor', size=(30,1), justification='center', background_color="#272533",
            text_color='white', font=('Franklin Gothic Book', 18, 'bold'))],
        [sg.Text('frame control:', size=(30,1), justification='left', background_color="#272533",
            text_color='white', font=('Franklin Gothic Book', 14, 'bold'))],
        [sg.Text('frame:0', size=(10,1), justification='center', background_color='black', text_color='red',
            font=('Digital-7',20), relief='sunken', key="_DISPLAY_FRAME_"),
        sg.Text('f.delta:10', size=(11,1), justification='center', background_color='black', text_color='blue',
            font=('Digital-7',20), relief='sunken', key="_DISPLAY_FRAME_DELTA_"),
        sg.Text('gender:male', size=(16,1), justification='center', background_color='black', text_color='green',
            font=('Digital-7',20), relief='sunken', key="_DISPLAY_GENDER_")],
        [sg.Button('prevFrame',**bt), sg.Button('nextFrame',**bt),
        sg.Button('playBackward',**bt), sg.Button('playForward',**bt)],
        [sg.Button('frameDelta-',**bt), sg.Button('frameDelta+',**bt)],
        [sg.Button('rotate',**bt),sg.Button('ResetBodyPose',**bt)],
        [sg.Text('coloring control', size=(30,1), justification='left', background_color="#272533",
            text_color='white', font=('Franklin Gothic Book', 14, 'bold'))],
        [sg.Button('noColor',**bt), sg.Button('colorByEffectedTriangles',**bt), sg.Button('colorByTrackVertexList',**bt)],
        [sg.Text('export files', size=(30,1), justification='left', background_color="#272533",
            text_color='white', font=('Franklin Gothic Book', 14, 'bold'))],
        [sg.Button('CalculateThetaWeights',**bt)],
        [sg.Text('load options:', size=(30,1), justification='left', background_color="#272533",
            text_color='white', font=('Franklin Gothic Book', 14, 'bold'))],
        [sg.Button('LoadPose',**bt),sg.Button('LoadShape',**bt),sg.Button('LoadBoth',**bt)],
        [sg.Button('Male',**bt),sg.Button('Female',**bt),sg.Button('Neutral',**bt),sg.Button('DefaultGender',**bt)],
        [sg.Text('pose information:', size=(100,1), justification='left', background_color="#272533")],
        get_all_keys_layout(),
        [sg.Button('Exit',**bt)],
    ]
    window: object = sg.Window('PyRenderOmer', layout=layout, element_justification='left', margins=(10,20), background_color="#272533", return_keyboard_events=True,finalize=True)
    return window

def table_param_entry_was_changed(element,values:dict,old_gui_values:dict)->bool:
    return element is not None and isinstance(element,sg.Input) and element.Key in values and element.Key in old_gui_values and values[element.Key] != old_gui_values[element.Key]

def table_animate_entry_was_pressed(event:str)->bool:
    try:
      param_name,num=get_entries_from_key_animate(key=event)
      return True
    except ValueError:
      return False


def playFramesAnimation(window:sg.Window,amass_plotter:AmassPlotter,amass_actor:AmassActor,delta_abs:int,playForward:bool)->None:
    delta_abs = abs(delta_abs)
    # assuming delta_abs>0
    animation_terminated=amass_actor.is_on_last_frame if playForward else amass_actor.is_on_first_frame
    delta = +delta_abs if playForward else -delta_abs
    while not animation_terminated():
        amass_actor.update_frame_with_frame_delta(frame_delta=+delta)
        amass_plotter.update_frame_from_amass_actor(amass_actor=amass_actor)
        update_GUI(window=window,amass_actor=amass_actor,delta=delta_abs)

def updateFrame(plotter:pv.Plotter,lastActor:vtkOpenGLActor,body_pose:BodyModel,frameID:int)->vtkOpenGLActor:
    plotter.remove_actor(lastActor)
    body_mesh = pv.wrap(human_mesh_utils.get_body_mesh_of_single_frame(body_pose,frameID))
    lastActor= plotter.add_mesh(body_mesh, show_edges=False)
    plotter.update()
    return lastActor


def get_vertex_area_vector(mesh:trimesh.Trimesh):
    area_faces = np.append(mesh.area_faces,0)
    vertex_faces_indecis = mesh.vertex_faces
    return np.sum(area_faces[vertex_faces_indecis],axis=1)/3

def calculate_theta_weight_wrap(window:sg.Window,amass_actor:AmassActor,amass_plotter:AmassPlotter)->None:
    gender = amass_actor.get_cur_gender()
    #frame = amass_actor.get_cur_frame
    #npz = amass_actor._cur_model_npz_file
    theta_dir_name='theta_vector_dir'
    fileName='theta_params_for_{}.pkl'.format(gender)
    full_file_path=os.path.join(os.curdir,theta_dir_name,fileName)
    calculate_theta_weight(window=window,amass_actor=amass_actor,amass_plotter=amass_plotter,fileName=full_file_path)


def calculate_theta_weight(window:sg.Window,amass_actor:AmassActor,amass_plotter:AmassPlotter,render_each_step:bool=True,save_theta_weight:bool=True,fileName:str='')->None:
    def value_gap_2():
        #hardcoded gap for now
        return np.array([-1,1])
    orginal_amass_actor = copy.deepcopy(amass_actor)
    base_amass_actor = copy.deepcopy(orginal_amass_actor)
    original_body_mesh = orginal_amass_actor.get_body_mesh()
    original_mesh_vertex_area_vector = get_vertex_area_vector(mesh=orginal_amass_actor.get_body_mesh())
    theta_weights = []
    cur_body_mesh = None
    #for param_name,_len in get_pose_param_names_and_lengths():
    for param_name,_len in [('pose_body',63)]:#uncommant when finish to debug
    #for param_name,_len in [('pose_body',2)]: #TODO for debug only!
        for i in range(_len):
            key=get_key_param(param_name,i)
            cur_theta_vector = np.zeros(len(original_body_mesh.vertices),dtype=bool)
            for param_value in value_gap_2():
                base_amass_actor.update_body_param_from_key_and_value(key=key,value=param_value)
                if render_each_step:
                    amass_plotter.update_frame_from_amass_actor(amass_actor=base_amass_actor)
                    update_GUI(window=window,amass_actor=base_amass_actor,delta=1) #delta don't have meaning on this phase 
                cur_body_mesh = base_amass_actor.get_body_mesh()
                vertex_moved_threshold=0.01
                vertices_that_moved=np.linalg.norm(cur_body_mesh.vertices-original_body_mesh.vertices,axis=1)>vertex_moved_threshold
                cur_theta_vector = np.logical_or(vertices_that_moved , cur_theta_vector)
            res = sum(original_mesh_vertex_area_vector[cur_theta_vector])
            theta_weights.append(res)
            print('cur key {},cur weight {}'.format(key,res))
            base_amass_actor = copy.deepcopy(orginal_amass_actor)
    refresh_actor=lambda:amass_plotter.update_frame_from_amass_actor(amass_actor=base_amass_actor)
    if render_each_step:
        refresh_actor()
        if amass_plotter.get_coloring_method()=='effected_triangles':
            #twice intentionaly
            refresh_actor()
    #normalized theta_weights

    theta_weights=torch.Tensor(theta_weights)
    theta_weights=theta_weights/theta_weights.sum()
    if save_theta_weight and fileName != '':
        save_obj(obj_to_save=theta_weights,f_name=fileName)
        #one can load with:
        """
        theta_weights=load_obj('theta_params_for_male.pkl')
        """


def value_gap():
    #hardcoded gap for now
    return np.arange(-1,1,0.2)
def animate_single_param(window:sg.Window,amass_plotter:AmassPlotter,amass_actor:AmassActor,key:str):
    param_name,num=get_entries_from_key_animate(key)
    original_param_value = copy.deepcopy(amass_actor.cur_body_params[param_name][amass_actor.get_cur_frame()][num])
    key=key.replace('&',':',1) # conver the delim
    def animation_step(param_value:float):
        amass_actor.update_body_param_from_key_and_value(key=key,value=param_value)
        amass_plotter.update_frame_from_amass_actor(amass_actor=amass_actor)
        update_GUI(window=window,amass_actor=amass_actor,delta=1) #TODO maybe propogate/pass the real delta here?


    #hardcoded gap for now
    for param_value in value_gap():
        animation_step(param_value=param_value)
        #amass_actor.update_body_param_from_key_and_value(key=key,value=param_value)
    animation_step(param_value=original_param_value)
    return


def openErrorMessege(error_str:str)->None:
    sg.theme("DarkTeal2")
    layout =  [[sg.Text(error_str)],[sg.Button("OK")]]
    window = sg.Window('My File Browser', layout, size=(500,100))
    while True:
        event, values = window.read()
        if event == sg.WIN_CLOSED or event=="OK":
            break
    window.close()
    return

def openFileDialog()->str:
    sg.theme("DarkTeal2")
    layout =  [[sg.Text("Choose a model file: ")],[sg.In(key="-IN-") ,sg.FileBrowse(file_types=(("Model file", "*.npz"),))],[sg.Button("Submit"),sg.Button("Exit")]]
    window = sg.Window('My File Browser', layout, size=(500,100))
    while True:
        event, values = window.read()
        if event == sg.WIN_CLOSED or event=="Exit":
            break
        elif event == "Submit":
            path_to_file = values["-IN-"]
            if os.path.exists(path_to_file) and human_mesh_utils.is_valid_npz_file(path_to_file):
                 window.close()
                 return path_to_file
            else:
                openErrorMessege("Error!\nFile:'{}'\nIs not a valid model file!".format(path_to_file))
    window.close()
    return ''

if __name__ == "__main__":
    openFileDialog()

