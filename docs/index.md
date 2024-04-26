# Neural rendering and X-ray
Exploring the potential for neural rendering methods in x-ray tomography

Author: Ivan Grega

## Objectives
X-ray computed tomography is an established method which can be used to obtain information about the internal composition of objects.
In this project, we try to answer the following two questions:
1. Can neural rendering methods (such as NeRF) be used to obtain similar information from a handful of projections (thereby reducing the required scanning time)?
2. Can such methods be used to calculate the deformation between two states of an object from a small number of projections?

## Progress to date 

### April 2024
#### Methods development
- created new method in nerfstudio (nerfstudio-xray)
- introduced volumetric loss
- enabled exporting as raw volume and stacks (nerfstudio/nerfstudio/scripts/exporter.py)
- custom dataparser to choose projections at regular angle intervals
- background color to white
- disabled scene contraction
- disabled scene scaling
- removed rgb head
- generated 4 main datasets: balls, pillars, cube, lattice
- captured experimental dataset

#### Theoretical analysis
We pose the problem of the inference of volume data statistically as entropy minimization problem.
One wants to minimize entropy of the data distribution $D$ conditioned on observing projections $P$.
```math
\begin{aligned}
D &\equiv \mathrm{volumetric \,data} \\
P &\equiv \mathrm{projection \,data} \\
H(D|P) &= H(D)-I(D;P)
\end{aligned}
```
Entropy is minimized when the mutual information between data and projeciton is maximised.
Suppose that projections $P$ are a collection of individual views $X_1,X_2,...$.
Then the mutual information can be decomposed as
```math
\begin{aligned}
I(D;X_z,X_2,X_3,...) &= I(D;X_1) + I(D;X_2|X_1) + I(D;X_3|X_1,X_2) + I(D;X_4|X_1,X_2,X_3) + ... 
\end{aligned}
```
We thus need to maximize the joint entropy of all projections $H(P)$.

Derived framework to calculate optimum angles with parallel beams and showed how these might change when perspective is added.

#### ML Results
Trained models to fit the four simple objects.
Depending on the degree of complexity of the object, different numbers of projections are needed for a satisfactory reconstruction.

### March 2024
#### Simple tests with synthetic data
First we created projections in Blender and tried to run out-of-the-box NeRF on it. It captures the simple geometry quite well from as few as 10 projections. See below the true object and the reconstruction.

![out_b](https://github.com/igrega348/neural_xray/assets/40634853/7df31aee-c164-471c-93a7-d374e0676f9f) ![out](https://github.com/igrega348/neural_xray/assets/40634853/041fd852-5951-47df-89c3-852737a0c20f)


#### Simple tests with real X-ray data
We managed to reconstruct four vertical bolts. See below an X-ray projection and the NeRF reconstruction. 

![ivan_nerf_1_0002](https://github.com/igrega348/neural_xray/assets/40634853/a9a1d096-ae68-497c-9b02-998e7d992fb6) ![out](https://github.com/igrega348/neural_xray/assets/40634853/ae1d0764-4b63-412b-ac6c-2e9aebcca5a3)

#### X-Ray rendering
We wrote a program in Go that lets us create synthetic X-ray projections for a variety of objects that can be reasonably conveniently represented as mathematical functions. Here’s an example of two such objects: a sphere with a cubic hole and a cube with a spherical hole.

![image](https://github.com/igrega348/neural_xray/assets/40634853/ccc4ba65-6578-41b3-82df-29ddbc6fa576)

The correctness of these synthetically-generated projections was verified using a Python tomography reconstruction package tomopy. Below are reconstructed cross-sections.

![image](https://github.com/igrega348/neural_xray/assets/40634853/1c3f33b6-bea5-4cd0-935b-88521768021a)
