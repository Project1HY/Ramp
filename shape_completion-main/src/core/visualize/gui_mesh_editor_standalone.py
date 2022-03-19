import pyvista as pv
import copy
from human_body_prior.body_model.body_model import BodyModel
#from visualizer import get_
import PySimpleGUI as sg
#based on https://github.com/israel-dryer/PyDataMath-II/blob/master/calculator_trinket.py
#from https://pysimplegui.trinket.io/demo-programs#/demo-programs/ti-datamath-ii-calculator
#from visualizer import get_body_mesh_of_single_frame
from human_mesh_utils import get_body_mesh_of_single_frame
import gui_utils
from vtkmodules.vtkRenderingOpenGL2 import vtkOpenGLActor

def gui_loop(amass_actor:gui_utils.AmassActor):
    cur_frame_delta=10
    amass_plotter = gui_utils.AmassPlotter()
    amass_plotter.show_interactive_mode()
    window=gui_utils.get_GUI()
    gui_utils.update_GUI(window=window,amass_actor=amass_actor,delta=cur_frame_delta)
    old_gui_values=dict()
    while True:
        event, values = window.read()
        element = window.find_element_with_focus()
        print(event)
        if event == 'Exit':
            break
        elif event == 'ResetBodyPose':
            amass_actor.reset_body_pose()
            amass_plotter.update_frame_from_amass_actor(amass_actor=amass_actor)
        elif event == 'prevFrame':
            amass_actor.update_frame_with_frame_delta(frame_delta=-cur_frame_delta)
            amass_plotter.update_frame_from_amass_actor(amass_actor=amass_actor)
        elif event == 'nextFrame':
            amass_actor.update_frame_with_frame_delta(frame_delta=+cur_frame_delta)
            amass_plotter.update_frame_from_amass_actor(amass_actor=amass_actor)
        elif event == 'playForward':
            gui_utils.playFramesAnimation(window=window,amass_plotter=amass_plotter,amass_actor=amass_actor,delta_abs=cur_frame_delta,playForward=True)
        elif event == 'playBackward':
            gui_utils.playFramesAnimation(window=window,amass_plotter=amass_plotter,amass_actor=amass_actor,delta_abs=cur_frame_delta,playForward=False)
        elif event == 'noColor':
            amass_plotter.set_coloring_method(coloring_method='none')
            amass_plotter.update_frame_from_amass_actor(amass_actor=amass_actor)
        elif event == 'colorByEffectedTriangles':
            amass_plotter.set_coloring_method(coloring_method='effected_triangles')
            amass_plotter.update_frame_from_amass_actor(amass_actor=amass_actor)
        elif event == 'CalculateThetaWeights':
            print('CalculateThetaWeights')
            gui_utils.calculate_theta_weight_wrap(window=window,amass_actor=amass_actor,amass_plotter=amass_plotter)
        elif event == 'frameDelta+':
            cur_frame_delta=min(cur_frame_delta+1,amass_actor.get_max_frame_num())
        elif event == 'frameDelta-':
            cur_frame_delta=max(cur_frame_delta-1,1)
        elif event == 'rotate':
            amass_plotter.rotate()
        elif event == 'LoadShape':
            new_model_npz_file= gui_utils.openFileDialog()
            if new_model_npz_file!= '':
                amass_actor.update_body_shape_from_model_npz_file(model_npz=new_model_npz_file)
                amass_plotter.update_frame_from_amass_actor(amass_actor=amass_actor)
        elif event == 'LoadPose':
            new_model_npz_file= gui_utils.openFileDialog()
            if new_model_npz_file!= '':
                amass_actor.update_body_pose_from_model_npz_file(model_npz=new_model_npz_file)
                amass_plotter.update_frame_from_amass_actor(amass_actor=amass_actor)
        elif event == 'LoadBoth':
            new_model_npz_file= gui_utils.openFileDialog()
            if new_model_npz_file!= '':
                amass_actor.load_new_model_from_model_npz_file(model_npz=new_model_npz_file)
                amass_plotter.update_frame_from_amass_actor(amass_actor=amass_actor)
        elif event == 'Male':
            print('male')
            amass_actor.update_gender('male')
            amass_plotter.update_frame_from_amass_actor(amass_actor=amass_actor)
        elif event == 'Female':
            print('female')
            amass_actor.update_gender('female')
            amass_plotter.update_frame_from_amass_actor(amass_actor=amass_actor)
        elif event == 'Neutral':
            print('Neutral')
            amass_actor.update_gender('neutral')
            amass_plotter.update_frame_from_amass_actor(amass_actor=amass_actor)
        elif event == 'DefaultGender':
            print('DefaultGender')
            amass_actor.reset_gender()
            amass_plotter.update_frame_from_amass_actor(amass_actor=amass_actor)

        # input bottoms
        elif gui_utils.table_param_entry_was_changed(element=element,values=values,old_gui_values=old_gui_values):
            if not gui_utils.isfloat(values[element.Key]):
                window[element.Key].update(old_gui_values[element.Key]) #dont change cell,unvalid input param.reset entry
            else:
                #we arrived to valid update.valid input
                print(values[element.Key])
                amass_actor.update_body_param_from_key_and_value(key=element.Key,value=values[element.Key])
                amass_plotter.update_frame_from_amass_actor(amass_actor=amass_actor)

        elif gui_utils.table_animate_entry_was_pressed(event=event):
            gui_utils.animate_single_param(window=window,amass_plotter=amass_plotter,amass_actor=amass_actor,key=event)

        """
        elif event == 'colorByTrackVertexList':
            amass_plotter.set_coloring_method(coloring_method='track_vertex_list')
            amass_plotter.update_frame_from_amass_actor(amass_actor=amass_actor)
        """
        #update values
        gui_utils.update_GUI(window=window,amass_actor=amass_actor,delta=cur_frame_delta)
        #save old values
        old_gui_values = values

