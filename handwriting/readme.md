# My Text in Your Handwriting

## Overview

This directory contains all of the code specific to my paper 'My Text in Your Handwriting' (with Oisin and Gabe). Be warned that there is an awful lot of automatic compilation required to run these scripts (Both proper C modules compiled with setup tools, as well as scipy.weave) - works almost out of the box on Linux, but other operating systems may prove harder.

Be aware that two different licenses have been used, as indicated below. Apache 2.0 for the stuff we don't mind being used freely (commercial use is OK), and GNU Affero GPL 3.0 for the bits we want to license (research use is OK, but commercial use would be extremely problematic). If you're unfamiliar with the Affero variant of the GPL then it strengths the GPL further, by forcing you to share the code with users even if they are the other side of a web service.


## Modules

`let (Apache 2.0)` - Line extraction tool; a GUI. Lets you prepare scanned in handwriting for synthesis; in addition to extracting the line includes tagging, handwriting recognition, alpha matting, indicating the rule of the page and a whole bunch of other things. Run with 'python main.py'.

`hst (GNU Affero GPL 3.0)` - Handwriting synthesis tool; the second GUI. Does exactly what you would expect - lets you synthesise handwriting. Contains all of the core synthesis code that brings the modules together to make it work, plus the compositing subsystem. Run with 'python main.py'.


`calibrate_printer (Apache 2.0)` - Does a closed loop colour calibration between a scanner and printer. For printing out synthetic handwriting to look as close as possible to real ink (which is often not that good - even good inkjets can't match a Biro!)

`corpus (Apache 2.0)` - Analyses the 100 most popular project Gutenberg books to generate typical statistics for the English language, then uses that to extract short blocks of text from the corpus for an author to write down in their handwriting. The text is selected to be representative of the English language, so we can learn a good model of their handwriting.


`line_graph (Apache 2.0)` - The data structure that represents the line extracted from text. Includes the majority of the C-optimised functionality of the handwriting system, and consequentially has some really strange features - absurdly feature rich in other words.

`recognition (Apache 2.0)` - Code to train a handwriting recognition model, as used by the line extraction tool for automatic tagging.


## Running

It's quite involved! Firstly, you don't install it, as it just runs directly out of the directory. Secondly, it's probably best to forget Windows - whilst I assume it can be done I've never tried, and expect it's a lot of work, even for me. I develop on Linux, so that's the only platform I can be sure of, but I know my code has been successfully run on Mac, often with only slight code changes. The below was written assuming Linux.

So, the two main tools can be found in the `helit/handwriting` directory:

 * `let` is the `line extraction tool` - you use it to tag a handwriting sample (after scanning it in).
 * `hst` is the `handwriting synthesis tool` - which does what it says it does!

There are other things in there as well, but I would ignore them for now.

My advice is to start by running something simpler than either of these two - I use automatic compilation in my code (very convenient when your doing research, and have a plan that changes every 5 minutes!), so we want to find out if that is working, noting that there are two types of automatic compilation in play.

First things first, check that python and numpy/scipy are installed. On the command line, type python - you should get a python environment. Then type 'import numpy' and check it worked. If the first fails you have a really weird distro - haven't come across a version of Linux that doesn't have python installed by default for some time! (this should be python 2 btw, not python 3). Not having numpy is however likely - you will want to install scipy using the package manager, as numpy is part of it now and you get both of them or neither of them. Be aware that it will need the dev version, as my code has to link with numpy. This will get tested below.

For the actual code I would start with the `helit/frf` directory - that's the random forest implementation it uses to do some machine learning stuff. Its a straight C module that automatically compiles itself in the 'traditional' way, so a good initial test. If you run one of the `test_*.py` files, either with `python` or `./`, see if it successfully compiles itself before running and writing out some text that looks like its doing something! (But don't run `test_vs_scikit_learn.py` - that requires scikit be installed, which is irrelevant as not used) If it doesn't work that means that python's setuptools package can't find a C compiler - typically installing `gcc` should be enough to fix that. Most distros these days have a package with a name like `buildessentials` which give you all of the standard compilers and related tools, such as make. There is also the possibility of it not being able to find either the python or numpy header files - if so you need to install dev versions of those packages.

Now we need to find out if the other compilation technique is working - odds are it is, as it also uses the compiler that setup tools finds, but just in case. For this I would run `test_simple.py` in `helit/svm` - it will kick out a bunch of warnings about using the old numpy api, but should print out `loo: ` when done. Note that this module isn't actually used by the system - its just a good test case of inline C++ compilation without anything else more complicated going on. This is testing scipy.weave - be warned that the most recent version of scipy have depreciated this package. You will need to install it seperately; it can be found here: https://github.com/scipy/weave To install try `pip install weave`.

Next thing to test is if the GUI code is working. Run `image_viewer.py` in `helit/utils_gui` - good chance it will error out, as it needs the python bindings for gtk3 to run. So install that from the package manager. Don't need to actually load an image, just make sure the boring grey GUI appears!

Now we are far enough along to actually try running the handwriting GUIs - in both cases you run 'main.py' in their respective directories. In both cases they should sit there compiling themselves for a while then show the GUI. Don't run one while the other is compiling itself - there are shared modules that could go screwy if two programs were trying to compile them at the same time! Actually using the system is rather involved, but I would start by trying to get synthesis working. From the http://visual.cs.ucl.ac.uk/pubs/handwriting/ web page you can download some handwriting samples that have already been tagged (The 2.1 gig file), and also the random forest models. Put the random forest models where the readme file says, though you only need cost_proxy.rf for now. Then run the synthesis GUI - the tab shown at the start is where you load the model. Note that we always split our model into many smaller files, each 2-3 lines of text, so the interface allows you to load multiple models and combine them (use shift select in the file browser; this is partly for breaking it down into small chunks when tagging, for ones sanity and keeping run time fast, but also because we needed to conduct experiments where we used all samples but one to generate the excluded sample). There is then a tab where you can enter text and the generate button in the middle, output image on the right. I won't give instructions for tagging yet - lets see if we can get this far first, but to give you an idea there is a video that shows tagging available: https://www.youtube.com/watch?v=TZqS3SBOzKE

When running hst some functionality dependends on having copied cost_proxy.rf into that directory (it can be obtained from the UCL project website).


## Tagging

Two warnings:
 * This is research software, so it's hardly stable or easy to use, and even if I wrote perfect instructions you might finish tagging some handwriting and find it doesn't work, with no easy way to determine why. Plus it's slow, and it takes a lot of time to prepare a handwriting sample.
 * For clarity, the 'font' it creates is for my own 'font engine' only, so you can't transfer it outside of my software stack - it won't work in any word processor for instance. Only with `hst`.

Anyway, here are some instructions for tagging a handwriting samples. Pretty involved I'm afraid!

 1. You need some words to write out. You can probably skip this step, and just write out some stuff - a pangram is always a good place to start. But if you want to do it 'properly' then first download the Project Gutenberg corpus (from http://visual.cs.ucl.ac.uk/pubs/handwriting/) and unzip it into `helit/handwriting/corpus`, so you now have the folder `helit/handwriting/corpus/data` containing lots of text files. Run the script `./make_db.py` in the corpus directory - it will generate the file `corpus.ply2` and print out a bunch of statistics about the data set. Now run `./make_sample.py`, in the same directory - it will sit there for a while and then print out a set of excerpts to write out. You may want to pipe the result into a file for printing/ease of viewing: `./make_sample.py > write_me.txt` Be warned that sometimes the sample isn't suitable - I left a few books in the dataset that contain small amounts of non-English text, and if the person writing out the sample is writing words they are unfamiliar with they keep pausing, which distorts their handwriting.

 2. Print out `lines.pdf`, attached. Now write out the words on the back of the printed paper. The idea is that you can see the lines, but they won't appear when scanned in. Also, write on every other line - you don't want ascenders and descenders to overlap. Be warned that the pen you choose can affect how hard it is to tag - I would avoid anything really cheap (As in a biro or cheap pencil. Its all down to line quality - you don't want one that has breaks or a lot of grain in it.).

 3. Scan the page(s) in. Aim for at least 300dpi, noting that 600dpi is preferred. Avoid saving to a `pdf` or `jpeg` - scanner software usually compresses everything heavily and introduces artefacts with these formats, which make the later stages harder. A `tiff` is usually the best option.

 4. The let software doesn't work too well if you give it really large images - I usually use an image editor to cut each page up into smaller files, each 2 or 3 lines long (written, so 4-6 actual printed lines). I also convert to png at this point, to be 100% sure that everything is lossless and no artefacts will be introduced.

 5. Load an image into let - the following has to be done for each image in turn. I would check things are working however, and test the generated files in hst as you go. First time you do this it can take quite a long time. I also use a graphics tablet (as in a device that allows you to use a special pen and 'draw' on your computer, not the comedy-sized mobile phones.) - the interface has been optimised for that scenario, but still works with a normal mouse.

So, the first thing to realise is that all of those buttons on the bottom bar of let do stuff, and you need to use about half of them whilst tagging. To make it really entertaining I didn't create my own icons and just used the best match from the theme I happened to have installed, so I would advise hovering over them to find out what they actually do, rather than trying to read anything into the image! The main set are two blocks of 5 buttons each in the middle of the bar - the first block chooses the interaction mode, the second which layers are visible. The image area uses rmb dragging to pan and the wheel to zoom btw, with all functions on the lmb, depending on the currently selected interaction mode. I would watch the interaction YouTube video for an overview of what I am about to explain: https://www.youtube.com/watch?v=TZqS3SBOzKE

 6. First you need to set the `rule homography`, that is tell the software where the lines you wrote on (which you can't see!) are. If you click the second interaction mode (looks like a set square ruler on some paper for me) the lines will now show as a grid - you can click-drag anywhere to move the grid around, the idea being to align the green horizontal lines with the written on lines. Be warned that this interaction mode is really unstable - tried to create a really flexible interface (any drag operation works), but it wasn't worth my time to implement it properly when that turned out to be kinda complicated! Best way to use it is to only click-drag 3 or 4 times, each one at an extreme of the text, as if all of your clicks are really close together it will probably be wrong at the extremes. If it does go wrong, and its really obvious when it has, the 'Reset Rule' option in the 'Algorithms' menu allows you to recover and try again.

 7. Now you need to separate the text from the background (referred to as thresholding) - click the second icon in the layers mode (looks like a pair of scissors for me), which will make it calculate which pixels are ink and which are paper. This will take a few seconds. Once done you need to inspect it and see if it messed up at all - I usually zoom in and look at each word, toggling the layer visibility so I can see if there has been anything that was masked as background when it should have been foreground. If there are mistakes you use the last three interaction modes - they are simply paint brushes that allow you to mark an area as ink (tick icon) or as background (cross icon), or remove any mark (paintbrush icon). Note that after painting on corrections you need to rerun the thresholding algorithm that ran automatically when you showed the thresholding layer - its the `threshold` menu item in the algorithms menu.

 8. Click the `infer alpha` button in the `File` menu - this will save a new copy of the image file, where the background has been removed.

 9. Click the 4th layer button, to show the extracted line - it will automatically extract it if you have not already done so from the menu. I would also recommend untoggling the thresholding layer, as its just distracting (also, the layer to the right of the thresholding is the corrections you paint on, which you might want to disable as well). Go over and check the line is reasonable, but I wouldn't worry too much about it.

 10. Now automatic tagging - this requires that the `hwf.rf` file be in the same directory, or a parent directory, of the image files you're working on. (Designed this way so I can use different models for different handwriting samples) Be warned that the automatic tagging model has been trained on a very limited set of data - its not very good, and if your handwriting is too different it might be quicker to just do it manually. If you run 'Auto tag' under algorithms an interface should come up (with no error I hope!), which will show each line of handwritten text with a text box you can type in below. Into each text box type (Or, if you generated text to write, c&p) the text above it. Leave text boxes for empty lines blank. Then hit go and wait a bit.

 11. Automatic tagging is never perfect, so you have to check it; this is all done in the first interaction mode, which is a pencil icon for me. Zoom in and hover the mouse over each letter in turn. It will show the letter in the text entry bar above the icons and also show a bounding box and the edges of the line that it thinks end the letter. The main mistake the automatic tagging does is not putting in the limits (I coded it to only add them if its sure, as its quicker to add them only rather than remove a bad guess and then add them back). Note that the first letter of the word is preceded by an underscore, and the last letter post-ceded by an underscore. Ligatures are automatically detected as the untagged lines that link tagged letters. The actual operations you can perform are:

 * Dragging across the ink line to split it, and create new segments.
 * Dragging between two segments to join them - this will delete any splits if they are connected (share one or more splits), or create a yellow link line if they are not. Link lines are usually used to connect the dot (tittle is the technical word!) of an 'i' to its stem.
 * Dragging across a link to remove it.
 * Typing while the mouse is hovered over a letter to change its tag. (you can actually type anything, and have multiple tags that are comma separated. The tag is assigned to the closest point to the mouse, so you can tag first, then split if you want.)

 12. Hit save. Done. Note there is also a `Save directory density` option - if you are tagging multiple files this will save some of the computation to a file in the directory, which it will automatically load each time you tag a file in that directory to save time and ensure consistency between all samples (this only really matters if you use a pen where you write faster than the ink comes out, so the line gets lighter across the page).

Note that whilst I presented the above as a fixed series of steps the software is actually capable of working in any order(-ish). This is particularly useful for going back and making corrections if you spot an error in hst that's explained by bad tagging. In fact, one way to use the software is to ignore my meticulous instructions for getting it all correct, just click through the automatic steps and then go back and make corrections as required. Also be aware that the system is fairly robust to all sorts of things - neither the thresholding or the extracted line have to be perfect for it to work.


## Printer Calibration

The colour calibration is closed loop, so in principal requires access to the original scanner if your using the files provided. Though this can be reasonably approximated by assuming all scanners are identical, as it is the printer that causes the most problems. The steps are:

 1. Run `handwriting/calibrate_printer/make_calibration_target.py` to create calibration_target.png
 2. Print the calibration target on the printer you intend to print the fake handwriting.
 3. Scan the target back in. This should be done on the scanner all of the handwriting samples were scanned in on, but it's not wholly unreasonable to use another scanner.
 4. Run `handwriting/calibrate_printer/calibrate.py` This will bring up a GUI to create the calibration file - you open the scanned colour target and then click to align the virtual grid to the scanned grid. Note that the colours of the corners should match the colours of the target - you might have rotated the target 180 degrees when scanning. When done save the calibration from the menu.
 5. You can now use the `handwriting/calibrate_printer/apply_cm.py` command line tool to convert original image files - if run with `-h` it will tell you how to use it, but it's simply `./apply_cm.py input_file.png file_saved_above.colour_map output_file.png` Note there is also `apply_cm_dir.py`, which can be used to process an entire directory at once.
 6. The file generated above is what you now send to the printer - it will have distorted the colours to make the printer do something closer to the right thing.

