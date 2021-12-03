Files used in the study: 
Evolution of rift systems and their fault networks in response to surface processes
Derek Neuharth, Sascha Brune, Thilo Wrona, Anne Glerum, Jean Braun, Xiaoping Yuan 


Parameter files are written in the format: (rift type)-(Kf value) 
e.g., the asymmetric model with a kf of 1e-5: asymmetric-1e-5.prm


Model runs in the study used dealii 9.2.0.
The aspect version used for this study is found at: 
https://github.com/Djneu/aspect/tree/fault_analysis

Additional ASPECT plugins written by Anne Glerum are used for the LAB perturbation, plastic strain noise, and refining the mesh based on compositions. 
These plugins are found in the "plugin file" folder.


This study utilizes the fatbox python toolbox written by Thilo Wrona, and found at:
https://github.com/thilowrona/fatbox

Scripts used to extract, correlate, and calculate fault properties are found in the "fatbox scripts" folder.
To analyze the faults, csv data must be extracted from paraview. An example state file is included (v. 5.9.0).
One script used to showcase the processing and create Figures 3, 5, and 7, is included.


To install ASPECT with FastScape.

1. Clone the ASPECT branch and a fastscape version.
	git clone -b fault_analysis https://github.com/Djneu/aspect
	git clone https://github.com/fastscape-lem/fastscapelib-fortran   
        *Note: study was done using FastScape commit 18f2588

2. Copy the Named_VTK.f90 file from the ASPECT directory into the FastScape source directory
	cp aspect/Named_VTK.f90 fastscapelib-fortran/src/

3. In the fastscape folder, add the following line to CMakeLists.
	set(FASTSCAPELIB_SRC_FILES
           ${FASTSCAPELIB_SRC_DIR}/Named_VTK.f90
	   )

4. Create a build directory for fastscape and compile it with an added flag for creating a shared library.
	cmake -DBUILD_FASTSCAPELIB_SHARED=ON /path/to/fastscapemake
	make


5. Compile ASPECT with a flag FASTSCAPE_DIR pointing to the fastscape build folder with the shared library
	cmake -DFASTSCAPE_DIR=/path/to/fastscape/build -DDEAL_II_DIR=/path/to/dealii -DCMAKE_BUILD_TYPE=Release /path/to/aspect
	make

        *Note: ASPECT will still compile and install even if a FastScape version is not found. During cmake, it should state whether or not
	       FastScape was found, if it was not you'll need to rerun cmake.
