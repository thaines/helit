My Text in Your Handwriting

This directory contains all of the code specific to my paper 'My Text in Your Handwriting' (with Oisin and Gabe). Each subdirectory is a different part of the system (which got absurdly large):


calibrate_printer - Does a closed loop colour calibration between a scanner and printer. For printing out synthetic handwriting to look as close as possible to real ink (which is often not that good - even good inkjets can't match a Biro!)

corpus - Analyses the 100 most popular project Gutenberg books to generate typical statistics for the English language, then uses that to extract short blocks of text from the corpus for an author to write down in their handwriting. The text is selected to be representative of the English language, so we can learn a good model of their handwriting.


line_graph - The data structure that represents the line extracted from text. Includes the majority of the C-optimised functionality of the handwriting system, and consequentially has some really strange features - absurdly feature rich in other words.

let - Line extraction tool; a GUI. Lets you prepare scanned in handwriting for synthesis; in addition to extracting the line includes tagging, handwriting recognition, alpha matting, indicating the rule of the page and a whole bunch of other things. Run with 'python main.py'.


recognition - Code to train a handwriting recognition model, as used by the line extraction tool for automatic tagging.
