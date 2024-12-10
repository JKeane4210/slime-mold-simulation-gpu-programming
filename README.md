# GPU Programming Final Project: CUDA C Slime Simulation

## Milwaukee School of Engineering | MS Machine Learning

### IS: GPU Programming | Fall Semester 2024

**Student:** Jonny Keane

**Advisor:** Dr. Sebastian Berisha

## Abstract

In this project, I implement a slime mold simulation in CUDA C that can be parallelized on a NVIDIA GPU. Slime molds are simple organisms that work effectively in colonies. Each particle has simple rules that dictate where to move and where to deposit chemoattract trails, so that other particles can choose to follow down explored paths. In the final state of this project, I achieve a kernel that can run 1 million particle threads operating in parallel and create a real time video feed that illustrates the evolution of the chemoattractant trails. Frames are rendered directly from GPU memory using the CUDA GL interop. Beyond this, I explored the use of blockers in the environment, food trails, and different hyperparameters for slime particle functionality to create an environment that allows for using an image as a food source, so slime particles may learn to converge to a state that takes on the shape of the given image. In addition to this, I also explore the use of having the particles leaving chemical trails of different colors based on different food sources they have consumed, which allowed for an even more interesting graphic to be created.

<img src="MSOE_Slime_FoodColoring.jpg">

An example of using different types of food to color the chemical trails of the particles.

<img src="MSOE_Slime.jpg">

An example of having food on the map for the particles to strive for, while also having dead regions that they cannot get to the food from.

## Building

On a computer with the nvcc compiler installed and having the glut libraries installed for using OpenGL (should most likely be preinstalled with CUDA driver), you can run the following command to compile:

```
nvcc slime_gpu_gl2.cu slime_kernels.cu -L libs -o slimeGL -lGL -lGLU -lglut
```

If you have prime-run set up on your computer (see resources below for help setting up), you can run the code with the following command:

```
prime-run ./slimeGL
```

## Resources

[Coding Adventure: Ant and Slime Simulations - YouTube](https://www.youtube.com/watch?v=X-iSQQgOd1A)

- This video was a main source of inspiration for doing this project. The experiments shown looked into a lot of other features such as use of different colors/species/hyperparameters.

[Characteristics of pattern formation and evolution in approximations of physarum transport networks](https://uwe-repository.worktribe.com/output/980579)

- This was a paper mentioned in the above video that talked about the specific implementation details of the slime particles. They also have a much more formalized exploration of different hyperparameters.

[NVIDIA - cuda-samples - GitHub](https://github.com/NVIDIA/cuda-samples)

- This repository has a lot of very helpful examples of doing different things in in CUDA. The main example I used in this project was from Samples/2_Concepts_and_Techniques/boxFilter, which was another visualization using the CUDA GL interop in a different application.

[prime-run Command Not Found - Stack Overflow](https://askubuntu.com/questions/1364762/prime-run-command-not-found)

- After executing the boxFilter code ended up not working, I found out I needed to use [prime-run](https://forums.developer.nvidia.com/t/getting-an-error-code-999-everytime-i-try-to-use-opengl-with-cuda/203769) command to run my executables with graphics code. I did not have this installed, however, but this Stack Overflow post helped me work through this.