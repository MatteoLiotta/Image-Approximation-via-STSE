# Image Approximation via STSE
> Image Approximation via Semi-Transparent Shape Evolution, OPTAI final project

![parisimage empty](https://github.com/user-attachments/assets/864a7e32-11a5-4a12-b1e9-af6da96de293)


## Project Overview

This project is a revisit of the “Genetic-lisa” Peter Braden’s project, which aims to recreate images with genetic algorithms., which had different implementations over time, from Python to C. In this case, you can find a modified python code in which not only explains the evolutionary procedure, but also tries to reduce computational limits due to image fitness evalutation, via multiprocessing. The aim in this case was also to increase generality by avoiding shape and color assumptions, making the same code capable of evolging different images dnas.

All original code was available [here](https://github.com/peterbraden/genetic-lisa/tree/master).

## Computational Resources

All the consideration power references should take into consideration that the project was realised entirely on a `MacBook Air M3 (2024) 256 GB` and `i7 Lenovo Legion`, with limited capabilities. Notice that:
* Mac computer has 8 cores
* Lenovo computer has 12 cores
  
and that's the reason of 8 and 12 patching inside the project code.

## Few words on evolutionary cicle and individuals

![cycle](https://github.com/user-attachments/assets/271bb820-b78a-4e50-afb3-23603f4c3dde)

Where the crossover and offspring generation originally proposed is:

![crossover](https://github.com/user-attachments/assets/bfaec362-28c0-4ce3-8612-7bc433456f3a)


## Differences and improvements

1. **Introduced a shape and color alteration through mutation**
  * Shape coloring mutation based on previous gene rather than randomic
    > ![image](https://github.com/user-attachments/assets/68a4fb97-5851-4ba2-8c5f-5a62b17cf4ce)
  
  * Shape size mutation based on previous gene rather than randomic
    > ![shapemutation](https://github.com/user-attachments/assets/60e3a990-2093-41e0-8fa8-39f3a8289be3)
    > A parameter can be used to set those min-max distances limits

2. **Better shape positioning**
  * Now it's more unlikely to observe absurde shape realization thanks to a better definition of space positioning

3. **Patching and multiprocessing**
  * Introduced patches
  * As a consequence
    * Increased precision on single-patches
    * Each core could handle evolution independently on a sub part of the image (patch)
    * Speed increase due to smaller comparison of images in fitness evaluation
      > each patch as a smaller reference (patched) compared to the original image, so the evaluation is computationally easier/faster
    * The DNA end up with `num_processes` more shapes, so more precision
    * Thanks to this solution, we enforce generality
      > No assumptions or reduced set of shapes and colors

3. **Fitness change towards faster computation**
   * We move from this fitness function to a cheaper one.
     ![lossfunctionoriginal](https://github.com/user-attachments/assets/69bb0f3e-7a64-4700-a8f7-7fc86942efbf)
 
4. **Introduced shape resising hyperparameter**
  * Instead of just fixed size shape we can fix the reduction ratio of shapes, so to have smaller shapes, leading to more details eventually captured

## Conclusion 

### Evolved image examples 

1. *Notre Dame Image*

![parisimage](https://github.com/user-attachments/assets/5431cb47-1004-4a40-8093-b0b5d7521345)

> Images evolved with all methods discussed: original code, shape-color mutation revisited and additional small-shape revisited evolution. Evolved over 50000 epochs, with 1 / 8 / 12 processes depending on the method used.
> *(Personal reference image)*

2. *Van Gogh Sunflowers*

![flowers](https://github.com/user-attachments/assets/6e6ef886-76ee-4fd8-b0d5-9f79df97a087)

> Image evolved with small-shape revisited evolution over 50000 epochs, with 12 processes. Reference: `[2]`

3. *B/W Picture of Trieste*

![trieste](https://github.com/user-attachments/assets/25f5647b-6685-4a36-96d4-80e8a328853d)

> Image evolved with small-shape revisited evolution over 50000 epochs, with 12 processes. `[2]`
>
> *(Personal reference image)*

## Project References

>
> `[1]` The evolution of a Smile, Peter Braden: https://github.com/peterbraden/genetic-lisa/ 
>
> `[2]` Mona Lisa Gif Evolution: https://github.com/peterbraden/genetic-lisa/blob/master/images/lisa-anim.gif 
>
> `[3]` Vase with Twelve Sunflowers (Arles, August 1888), Van Gogh. Neue Pinakothek, Munich: https://commons.wikimedia.org/wiki/File:Vincent_Willem_van_Gogh_128.jpg

