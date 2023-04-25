# -*- coding: utf-8 -*-
"""
Created on Tue Apr 25 13:18:44 2023

Visualisation code

@author: Jelmer
"""

import pickle
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Patch
from data_preprocessor import PreprocessData
import os

if __name__ == "__main__": 
    
    print("Importing input dataset.")
    B = "B.pickle"
    R = "R.pickle"

    B_data = PreprocessData(B).process(report=False)
    R_data = PreprocessData(R).process(report=False)
    
    print("Importing optimal solution values.")
    path = os.getcwd()
    outputs = os.listdir(f"{path}/results")
    
    with open("results/bins_used.pickle", 'rb') as handle:
        bins_used = pickle.load(handle)
        
    with open("results/I_info_solution.pickle", 'rb') as handle:
        item_info = pickle.load(handle)
        
    with open("results/Items_in_Bin.pickle", 'rb') as handle:
        items_in_bin = pickle.load(handle)
    
    if "bin_figures" not in os.listdir(f"{path}/results"):
        os.mkdir("results/bin_figures")
        
    labels = ["Regular Item", "Fragile", "Perishable", "Radioactive"]
    colors = ["cyan", "white", "green", "red"]
    hatchings = ["", "xx", "", ""]
    
    print("Creating plots.")
    
    width_plots = max(B_data["L"])
    height_plots = max(B_data["H"])
    
    for b in bins_used:
        fig, ax = plt.subplots()
        
        # Creating single dot (needed for some reason)
        ax.plot([0],[0])
        
        # Obtain bin info
        bl = B_data.loc[b, "L"]
        bh = B_data.loc[b, "H"]
        
        # Draw bin
        ax.add_patch(Rectangle((0,0), bl, bh,
                               edgecolor = 'blue',
                               fill=False,
                               lw=3))
        
        for item in items_in_bin[b]:
        
            # Obtain item info
            x = item_info[item][0]
            z = item_info[item][1]
            l = item_info[item][2]
            h = item_info[item][3]
            
            # Fragile or not:
            if R_data.loc[item, "f"]:
                hatching = hatchings[1]
            else:
                hatching = ""
                
            # Perishable, radioactive or neither:    
            if R_data.loc[item, "rho"]:
                color_id = 2
            elif R_data.loc[item, "phi"]:
                color_id = 3
            else:
                color_id = 0
            
            # Draw item
            ax.add_patch(Rectangle((x, z), l, h,
                         edgecolor = 'orange',
                         facecolor = colors[color_id],
                         fill=True,
                         hatch=hatching,
                         lw=2,
                         alpha=0.7))
            
            ax.annotate(item, (x+l/2, z+h/2), color='w', weight='bold', 
                fontsize=12, ha='center', va='center')
            
            
        # Draw bin
        ax.add_patch(Rectangle((0,0), bl, bh,
                               edgecolor = 'blue',
                               fill=False,
                               lw=3,
                               label="Bin Border"))
        
        # Defining legend
        handles = [
        Patch(facecolor=color, label=label, hatch=hatching, alpha=0.7) 
        for label, color, hatching in zip(labels, colors, hatchings)
        ]
        ax.legend(handles=handles)
        
        # Defining additional variables
        plt.title(f"Bin {b}")
        plt.xlabel("x")
        plt.ylabel("z")
        
        plt.xlim(0, width_plots)
        plt.ylim(0, height_plots)
        ax.set_aspect("equal", anchor = "C")

        plt.show()
        
        plt.savefig(f"results/bin_figures/bin{b}.pdf",bbox_inches='tight')
        
    print("Plots created.")