Line Graph

The key data structure used to contain handwriting - an arbitrary graph of lines with attached information, including texture coordinates, line details (radius, density, weight) and tags/splits to break it up into individual glyphs, labelled with the letters they represent.

Includes most of the core functionality of the handwriting project, and can do rather a lot. This includes cutting out segments (glyphs), applying homographies, distorting ligatures, multiple feature vector calculation methods and plenty of other stuff. Also has the all of the tests required to quickly query which part a user clicked on and which bit is visible, to support both rendering and interacting in a GUI.

Additionally includes layers for the utils_gui system that allow you to render the line graph, one for the line and one for the splits, links and bounding box around the segment closest to the mouse cursor.

Finally, viewer.py is a simple GUI for looking at a line graph file.

If you are reading readme.txt then you can generate documentation by running make_doc.py


Contains the following key files:

line_graph.py - Provides the LineGraph object, that being the point of this module. Will compile it if required.

line_layer.py - A rendering layer that draws the line of a LineGraph; supports multiple rendering modes.
line_overlay_layer.py - A rendering layer that adds splits, links and a bounding box to the above, but only for the segment nearest the mouse cursor.

viewer.py - Straight forward GUI for looking at line graph files.

test.py - A bit of unit testing.

readme.txt - This file, which is included in the html documentation.
make_doc.py - Builds the html documentation.

