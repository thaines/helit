utils_gui

Bunch of helpers for making GUIs - basically just a Viewer widget for use in GTK that provides a suitably sexy image viewer, with zooming, overlays etc.


viewer.py - The actual viewer object, to be used as a widget.

viewport_layer.py - Some helper classes used by the Viewer object, most importantly the Layer interface that everything that can draw to the image has to impliment.


tile_image.py - Layer object for use with a Viewer that draws an actual image; can do alpha matting.

tile_mask.py - Layer object that draws a colour for each label in a binary mask; alpha is supported.

tile_value.py - Layer object that does the same as the above but for an arbitrary set of labels.

reticle_overlay.py - Draws a reticle over the image, identifying the center.


readme.txt - This file, which is copied into the html documentation.

make_doc.py - Code that generates the html documentation.

