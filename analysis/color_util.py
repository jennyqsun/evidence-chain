import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import random
import colorsys
##### equations ################
def filter_light_colors(colors, lightness_threshold=0.7):
    filtered_colors = []
    for color in colors:
        # Ignore the alpha channel by slicing the first three elements (RGB)
        r, g, b, a = color
        # Convert RGB from 0-1 range to 0-255 range, if needed, but here it is already in 0-1
        # r, g, b = [x * 255 for x in (r, g, b)]
        
        # Convert RGB to HLS
        h, l, s = colorsys.rgb_to_hls(r, g, b)
        
        # Check if the lightness is below the threshold
        if l < lightness_threshold:
            filtered_colors.append(color)
    
    return filtered_colors



def get_color_list():
    # Choose a colormap that has a good range of distinct colors
    colormap1 = cm.get_cmap('tab20', 20)  # 'tab20' has exactly 20 distinct colors

    # Generate 20 distinct colors from the colormap
    colors1 = [colormap1(i) for i in range(colormap1.N)]
   
    # Choose a colormap that has a good range of distinct colors
    colormap2 = cm.get_cmap('Dark2', 8)  # 'tab20' has exactly 20 distinct colors

    # Generate 20 distinct colors from the colormap
    colors2 = [colormap2(i) for i in range(colormap2.N)]

    # Choose a colormap that has a good range of distinct colors
    colormap3 = cm.get_cmap('Accent', 8)  # 'tab20' has exactly 20 distinct colors

    # Generate 20 distinct colors from the colormap
    colors3 = [colormap3(i) for i in range(colormap3.N)]

    # Choose a colormap that has a good range of distinct ccolors
    colormap4 = cm.get_cmap('Paired', 12)  # 'tab20' has exactly 20 distinct colors

    # Generate 20 distinct colors from the colormap
    colors4 = [colormap4(i) for i in range(colormap4.N)]

    # Choose a colormap that has a good range of distinct ccolors
    colormap5 = cm.get_cmap('Set1', 12)  # 'tab20' has exactly 20 distinct colors

    # Generate 20 distinct colors from the colormap
    colors5 = [colormap5(i) for i in range(colormap5.N)]

    colors = colors1 + colors2 +colors3 + colors4 +colors5

    unique_colors = list(set(colors))
    # Applying the filter
    filtered_colors = filter_light_colors(unique_colors)
    random.seed(200) 
    random.shuffle(filtered_colors)
    return filtered_colors

