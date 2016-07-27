Recognition

This trains the handwriting recognition model used by the line extraction tool to automatically tag data. The first thing to note about this is it was never the focus of the paper - it remains a quick and dirty model, that worked well enough to save some time. I could list off a dozen improvements without thinking about it. Secondly, the parameter optimisation maximises the oob error of the random forest, rather than using a proper train/test split - I am fully aware that this is a bad idea, but again, this was never the focus and it works well enough. Besides, this model is used to position splits, not identify letters, so its the wrong objective function anyway. And the hyper-parameter optimisation algorithm is stupid.


You train a model for a directory of line_graphs, including all line_graphs found in subdirectories as well. Ultimately, it outputs hwr.rf in the directory. When the auto tagger of the line extraction tool is run it searches for this file, starting in the directory of the line_graph being tagged and then working up through each directory in turn, until it finds it or reaches the root and gives up. In other words, hwr.rf is written to the correct location for let to then use it.

The workflow is as follows:

Run ./optimise.sh <directory of line graphs>. This will create and populate runs.json, a list of parameter settings followed by oob score. When you have got bored/decided your computer has had enough stop this script. Note that runs.json is locked for writing, so you should run as many copies of optimise.sh as you have cores.

Run ./best.py It will print out the parameters for the best performing parameters tried. Copy those parameters into main.py, at the top. Note that the version of main.py checked into the repository has the parameters selected with a fairly sizeable run, so you might want to skip to the next step.

Run ./main.py <directory of line graphs>. This will learn a model with the (~best if above steps performed) parameters and save the hwr.rf file.

