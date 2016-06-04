Calibrate Printer

A small set of tools for doing a closed-loop colour calibration of a printer-scanner pair. Idea is you print the calibration_target.png file (created by make_calibration_target.py) on a printer, then scan it in. The program can then distort other images that have been scanned in using the same scanner so they print as close to correct on that same printer.

The tools are actually rather more general than the above idea, but the interface is just designed with that scenario in mind - you could introduce proper colour calibration targets with minimal effort, and chain colour calibrations as well. Given a colour calibration target and a camera you could actually work out the calibration of a computer monitor! (Nowhere near as good as a proper colorimeter of course. Can't even describe it as a 'poor mans' version, as proper colour calibration targets are expensive! Plus the tool chain has no way to actually apply it.) On the subject of the interface its a GUI for doing the calibration (so you can click to align the calibration targets - a printed-then-scanned one won't appear in a predictable position.), plus some command line scripts for then applying the calibration and distorting files ready for printing.

Internally its just mapping one 3D function to another, based on a set of known matches (from the calibration target and printed-then-scanned calibration target), with thin plate splines to interpolate, one per colour channel. It should be no surprise that most printers suck, and that you will struggle to get the dynamic range of real ink out of them. Inkjets are definitely the best choice if you care about realism more than being driven nuts by a technology that seems designed to induce insanity.

Scripts:

make_calibration_target.py - Generates calibration_target.png, which you print out then scan back in.

calibrate.py - GUI for doing calibration and saving colour_map files. Typical usage is to first load the scan, then click to align the homography (mouse wheel to zoom in, right mouse to pan. Also works with a graphics tablet!). Its generally wise to switch view between the (default) scan view and calibration target view, to check that there is no rotation (its easy to scan the page upside down!). This function is in the view menu, where you can also find a rotate function for fixing such problems easily - note that each corner of the homography has a unique colour to help with this. Then save the colour map.

visualise.py - Converts a colour_map into an image of pairs of colour swatches - good for sanity checking the output.

apply_cm.py - Applies a colour map to a single image - has a lot of command line options.

apply_cm_dir.py - Applies a colour map to an entire directory of images, saving the distorted images into a new directory with the same names.

