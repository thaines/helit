Line Graph

The key data structure used to contain handwriting - an arbitrary graph of lines with attached information, including texture coordinates, line details (radius, density, weight) and tags/splits to break it up into individual glyphs, labelled with the letters they represent.

Includes most of the core functionality of the handwriting project, and can do rather a lot. This includes cutting out segments (glyphs), applying homographies, distorting ligatures, multiple feature vector calculation methods and plenty of other stuff. Also has the all of the tests required to quickly query which part a user clicked on and which bit is visible, to support both rendering and interacting in a GUI.

If you are reading readme.txt then you can generate documentation by running make_doc.py


Contains the following key files:

line_graph.py - Provides the LineGraph object, that being the point of this module. Will compile it if required.

test.py - A bit of unit testing.

readme.txt - This file, which is included in the html documentation.
make_doc.py - Builds the html documentation.

