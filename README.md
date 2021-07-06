# objectiveMapping
functions to combine different FRF surveys into an integrated topo map
`
`objMapInterp.py` is responsible for the actual interpolation
`objMapPlots.py` has plotting tools for this library
`objMapPrep.py` has pre-processing routines in here

`FRF_duneLidarDEM_bathy_interp.py` is a single instance work flow. maybe a good place to start to learn

`operation_FRF_duneLidar_DEM_bathy_interp.py` is a work flow setup to batch process.  This relys on other CMTB libraries, and will take some supporting functions, none of which are pip installable. 
