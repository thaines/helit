My Text in Your Handwriting

This directory contains all of the code specific to my paper 'My Text in Your Handwriting' (with Oisin and Gabe). Each subdirectory is a different part of the system (which got absurdly large):


calibrate_printer - Does a closed loop colour calibration between a scanner and printer. For printing out synthetic handwriting to look as close as possible to real ink (which is often not that good - even good inkjets can't match a Biro!)

corpus - Analysis the 100 most popular project Gutenberg books to generate typical statistics for the English language, then uses that to extract short blocks of text from the corpus for an author to write down in their handwriting. The text is selected to be representative of the English language, so we can learn a good model of their handwriting.

