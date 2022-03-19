import matplotlib.pyplot as plt
import numpy as np
import os
#creating bar plots
#TODO add saving option
def save_or_show_fig_if_neccery(my_fig:plt,show:bool=True,path_to_save_fig:str='')->None:
    if show:
        my_fig.show()
    if path_to_save_fig != '':
        my_fig.savefig(path_to_save_fig,dpi=1200)
    plt.close(my_fig)

def get_bar_chart_with_x_labels(x_labels:list,data_bar:list,y_label:str,title:str,width_of_each_bar:float=0.35,show:bool=True,path_to_save_fig:str='',x_labels_font_size:int=10)->None:
    """
    see example at 'https://matplotlib.org/stable/gallery/lines_bars_and_markers/barchart.html#sphx-glr-gallery-lines-bars-and-markers-barchart-py'
    label - the label means
    """
    x = np.arange(len(x_labels))
    fig,ax = plt.subplots()
    bar = ax.bar(x,data_bar,width_of_each_bar)
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_ylabel(y_label)
    ax.set_xticklabels(x_labels,rotation='vertical')
    """
    for bar in list_of_bars:
        ax.bar_label(bar,padding=9)
    """
    fig.tight_layout()
    plt.setp(ax.get_xticklabels(), fontsize=x_labels_font_size, rotation='vertical')
    save_or_show_fig_if_neccery(my_fig=fig,show=show,path_to_save_fig=path_to_save_fig)
    """
    if show:
        plt.show()
    """

def get_grouped_bar_charts_with_x_labels(x_labels:list,data_label_bar_1:str,data_label_bar_2:str,data_bar_1:list,data_bar_2:list,y_label:str,title:str,width_of_each_bar:float=0.35,show:bool=True,path_to_save_fig:str='')->None:
    """
    see example at 'https://matplotlib.org/stable/gallery/lines_bars_and_markers/barchart.html#sphx-glr-gallery-lines-bars-and-markers-barchart-py'
    label - the label means
    """
    x = np.arange(len(x_labels))
    fig,ax = plt.subplots()
    list_of_bars = []
    list_of_bars.append(ax.bar(x-width_of_each_bar/2,data_bar_1,width_of_each_bar,label=data_label_bar_1))
    list_of_bars.append(ax.bar(x+width_of_each_bar/2,data_bar_2,width_of_each_bar,label=data_label_bar_2))
    ax.set_ylabel(y_label)
    ax.set_title(title)
    ax.set_xticks(x)
    ax.legend()
    ax.set_xticklabels(x_labels,rotation='vertical')
    """
    for bar in list_of_bars:
        ax.bar_label(bar,padding=9)
    """
    fig.tight_layout()
    save_or_show_fig_if_neccery(my_fig=fig,show=show,path_to_save_fig=path_to_save_fig)

