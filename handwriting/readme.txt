My Text in Your Handwriting

This directory contains all of the code specific to my paper 'My Text in Your Handwriting' (with Oisin and Gabe). Be warned that there is an awful lot of automatic compilation required to run these scripts (Both proper C modules compiled with setup tools, as well as scipy.weave) - works almost out of the box on Linux, but other operating systems may prove harder.

Be aware that two different licenses have been used, as indicated below. Apache 2.0 for the stuff we don't mind being used freely (commercial use is OK), and GNU Affero GPL 3.0 for the bits we want to license (research use is OK, but commercial use would be extremely problematic). If you're unfamiliar with the Affero variant of the GPL then it strengths the GPL further, by forcing you to share the code with users even if they are the other side of a web service.

Each subdirectory is a different module in the system (which got absurdly large):


let (Apache 2.0) - Line extraction tool; a GUI. Lets you prepare scanned in handwriting for synthesis; in addition to extracting the line includes tagging, handwriting recognition, alpha matting, indicating the rule of the page and a whole bunch of other things. Run with 'python main.py'.

hst (GNU Affero GPL 3.0) - Handwriting synthesis tool; the second GUI. Does exactly what you would expect - lets you synthesise handwriting. Contains all of the core synthesis code that brings the modules together to make it work, plus the compositing subsystem. Run with 'python main.py'.


calibrate_printer (Apache 2.0) - Does a closed loop colour calibration between a scanner and printer. For printing out synthetic handwriting to look as close as possible to real ink (which is often not that good - even good inkjets can't match a Biro!)

corpus (Apache 2.0) - Analyses the 100 most popular project Gutenberg books to generate typical statistics for the English language, then uses that to extract short blocks of text from the corpus for an author to write down in their handwriting. The text is selected to be representative of the English language, so we can learn a good model of their handwriting.


line_graph (Apache 2.0) - The data structure that represents the line extracted from text. Includes the majority of the C-optimised functionality of the handwriting system, and consequentially has some really strange features - absurdly feature rich in other words.

recognition (Apache 2.0) - Code to train a handwriting recognition model, as used by the line extraction tool for automatic tagging.
