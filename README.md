# mungus
A screenshot aligner that stitches images as you walk through a 2d game. Currently uses [BRIEF](https://doi.org/10.1007/978-3-642-15561-1_56) algorithm to perform pairwise image alignment.

![Example alignment](/images/example_alignment.png)

When running continuously, the aligner can consecutively stitch full resolution screenshots fast and accurately enough to recreate the entire map in Among Us:

![Example stitching](/images/full_map_stitch.png)

