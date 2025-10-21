Oct 19

Simple example of initializing with an input profile
NOTE: please check bugs
To start, copy coordinate.hpp to master branch athena/src/coordinate
copy pgen file inject_wind.cpp to master branch athena/src/pgen

The pgen reads in init_x1v.txt, init_rho.txt, init_temp.txt, init_vel.txt and initialzie domain with
these density, temperature and velocity. The init_*.txt file has the shape of total cell number.

(?) In principle it should work with AMR

The pgen has a comment-out gravity source term EnvGravity, it calculate enclosed mass and add 
gravity to each shell based on the mass. It is a simple implementation of self-gravity in spherical
coordinate. This implementation will not work with AMR.

--------
The athinput.csm is an example input file
The /plot directory has a python script wind_unit calculate physical unit, radiation simulation cannot rescale to arbitrary unit.

