# Simulation parameters

## Integrator
DiSECt uses semi-implicit Euler integration with a fixed time step.

| Parameter  | Unit | Description |
| ------------- |:----:| ------------- |
| `sim_duration` | $s$ | Duration of a simulation to roll out when calling `CuttingSim.simulate()` |
| `sim_dt` | $s$ | Integration time step |
| `sim_substeps` | | Number of steps to skip when populating the knife force profile |

## Material properties
The object being cut is simulated through the Finite Element Method (FEM). The following parameters define the behavior of the deformable mesh simulation:

| Parameter  | Unit | Description |
| ------------- |:----:| ------------- |
| `young`  | $N / m^2$ |  Young's modulus $E$ of the material being cut  |
| `poisson`  |  | Poisson's ratio $\nu$ of the material being cut  |
| `mu`, `lambda` | $N / m^2$ | Lamé parameters $\mu$, $\lambda$ of the material being cut; will be calculated automatically from the Young's modulus and Poisson's ratio via `convert_lame()` in cutting/utils.py |
| `density` | $kg / m^3$ | Density of the material being cut; the particles are assigned their masses based on the volume of the tetrahedra they are adjacent to |
| `damping` | | Damping applied in the FEM simulation of the material being cut |
| `minimum_particle_mass` | $kg$ | Minimum mass of each particle (zero by default, helps in some situations when the tetrahedral mesh is of poor quality) |
| `static_vertices`: `{"include": [], "exclude": []}` | | Define bounding boxes around particles which should be treated as static vertices (their masses are set to zero so that they serve as static boundary conditions to the mesh). Bounding boxes can be inclusive and exclusive, i.e. first the vertices are selected that lie in any of the bounding boxes defined under `"include"`, then those vertices are ignored which lie in any of the bounding boxes defined under `"exclude"`. The bounding boxes are defined as dictionaries `{"min": [x, y, z], "max": [x, y, z]}`. |

## Knife geometry
The knife's geometry is represented by a signed distance function (SDF), which is parameterized as follows:

| Parameter  | Unit | Description |
| ------------- |:----:| ------------- |
| `sdf_radius`  | $m$ | Radius around the knife's SDF causing contact to occur sooner than the actual knife geometry, but results in smoother contact normals  |
| `knife_sdf`.`spine_dim` | $m$ | |
| `knife_sdf`.`edge_dim` | $m$ | |
| `knife_sdf`.`spine_height` | $m$ | |
| `knife_sdf`.`tip_height` | $m$ | |
| `knife_sdf`.`depth` | $m$ | |

## Ground contact dynamics parameters
Parameters for the spring-damper contact model of the ground plane on which the mesh to be cut rests. Mesh vertices are treated as spheres with radius `ground_radius` and the ground is treated as a half space with its normal pointing upwards ($\begin{bmatrix}0 & 1 & 0\end{bmatrix}$).

| Parameter  | Unit | Description |
| ------------- |:----:| ------------- |
| `ground`  | Boolean | Whether to simulate the ground (unless static boundary conditions are defined the mesh will free-fall) |
| `ground_ke` | | Contact normal force stiffness |
| `ground_kd` | | Contact damping |
| `ground_kf` | | Contact friction stiffness |
| `ground_mu` | | Friction coefficient $\mu$ |
| `ground_radius` | $m$ | Radius of the mesh vertices to be considered as spheres |

## Knife contact dynamics parameters
The contact dynamics of the knife interacting with the material is simulated through an SDF-edge contact model where the knife geometry is defined by a signed distance field (SDF) which collides with the cutting springs that are treated as edges in the mesh. The following parameters are replicated with the prefix `surface_` to indicate special settings that apply only to the cutting springs at the outer layer of the mesh, to allow for the simulation of harder skins in fruits and vegetables, for example.

| Parameter  | Unit | Description |
| ------------- |:----:| ------------- |
| `sdf_ke` | | Contact normal force stiffness |
| `sdf_kd` | | Contact damping |
| `sdf_kf` | | Contact friction stiffness |
| `sdf_mu` | | Friction coefficient $\mu$ |

## Cutting spring parameters
These parameters pertain to the cutting springs that connect the mesh halves and are weakened in proportion to the force the knife exerts on them. Again, these parameters are replicated with the prefix `surface_` to indicate special settings that apply only to the cutting springs at the outer layer of the mesh.

| Parameter  | Unit | Description |
| ------------- |:----:| ------------- |
| `cut_spring_ke` | | Spring stiffness |
| `cut_spring_kd` | | Spring damping |
| `cut_spring_softness` | | Softness coefficient $\gamma$ used in the linear spring weakening model |
| `cut_spring_rest_length` | $m$ | Rest length of the spring (zero by default) |