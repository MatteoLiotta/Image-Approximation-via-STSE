# Image Approximation via STSE
> Image Approximation via Semi-Transparent Shape Evolution, OPTAI final project

![parisimage empty](https://github.com/MatteoLiotta/Image-Approximation-via-STSE/blob/main/Presentation/Images/parisimage%20empty.png)


## Project Overview

This project is a revisit of the “Genetic-lisa” Peter Braden’s project, which aims to recreate images with genetic algorithms. It has seen different implementations over time, from Python to C. This version includes a modified python code which not only explains the evolutionary procedure, but also tries to reduce computational limits due to image fitness evalutation via multiprocessing. The goal was was also to increase generality by avoiding shape and color assumptions, making the same code capable of evolging different images DNAs.

All original code was available [here](https://github.com/peterbraden/genetic-lisa/tree/master).

## Computational Resources

All computational considerations are based on the following two machines the consideration power references should take into consideration that the project was realised entirely on a  and , with limited capabilities. Notice that:
* `MacBook Air M3 (2024) 256 GB`, with `8 cores CPU`
* `i7 Lenovo Legion`, with `12 cores CPU`
  
That justifies the idea behind 8 and 12 patching number inside the project code.

## Few words on evolutionary cicle and individuals

![cycle](https://github.com/MatteoLiotta/Image-Approximation-via-STSE/blob/main/Presentation/Images/cycle.png)

Where the crossover and offspring generation originally proposed is:

![crossover](https://github.com/MatteoLiotta/Image-Approximation-via-STSE/blob/main/Presentation/Images/crossover.png)


## Differences and improvements

1. **Introduced a shape and color alteration through mutation**
  * Shape coloring mutation based on previous gene rather than randomic
    > ![image](https://github.com/MatteoLiotta/Image-Approximation-via-STSE/blob/main/Presentation/Images/colorpalette.png)
  
  * Shape size mutation based on previous gene rather than randomic
    > ![shapemutation](https://github.com/MatteoLiotta/Image-Approximation-via-STSE/blob/main/Presentation/Images/shapemutation.png)
    > A parameter can be used to set those min-max distances limits

2. **Better shape positioning**
  * Now it's more unlikely to observe unrealistic shape realization thanks to a better definition of space positioning

3. **Patching and multiprocessing**
  * Introduced patches idea, with these consequences:
    * Increased precision on single-patches
    * Each core could handle evolution independently on a sub part of the image (patch)
    * Speed increase due to smaller comparison of images in fitness evaluation
      > each patch as a smaller reference (patched) compared to the original image, so the evaluation is computationally easier/faster
    * The DNA end up with `num_processes` more shapes, so more precision
    * Thanks to this solution, we enforce generality
      > No assumptions or reduced set of shapes and colors

3. **Fitness change towards faster computation**
   * We move from the original fitness function to a cheaper one.
 
4. **Introduced shape resising hyperparameter**
  * Instead of just fixed size shape we can fix the reduction ratio of shapes, so to have smaller shapes, leading to more details eventually captured

## Conclusion 
The project demonstrates the evolution of Semi-Transparent Shapes with, showing the capability of capturing and recreaing details and characteristics of a selected reference image over time. It's important to underline that this specific implementation mainly focuses over both speed and generality, trying to find a trade-off between them while recreating an accurate image approximation.

It's important to notice that different an more playful results could be expected to be achieved by with the introduction of different shapes or by fixing just some of them.  ;)

### Evolved image examples 

1. *Notre Dame Image*

![parisimage](https://github.com/MatteoLiotta/Image-Approximation-via-STSE/blob/main/Presentation/Images/parisimage.png)

> Images evolved with all methods discussed: original code, shape-color mutation revisited and additional small-shape revisited evolution. Evolved over 50000 epochs, with 1 / 8 / 12 processes depending on the method used.
> *(Personal reference image)*

2. *Van Gogh Sunflowers*

![flowers](https://github.com/MatteoLiotta/Image-Approximation-via-STSE/blob/main/Presentation/Images/flowers.png)

> Image evolved with small-shape revisited evolution over 50000 epochs, with 12 processes. Reference: `[3]`

3. *B/W Picture of Trieste*

![trieste](https://github.com/MatteoLiotta/Image-Approximation-via-STSE/blob/main/Presentation/Images/trieste.png)

> Image evolved with small-shape revisited evolution over 50000 epochs, with 12 processes.
>
> *(Personal reference image)*

## Project References

>
> `[1]` The evolution of a Smile, Peter Braden: https://github.com/peterbraden/genetic-lisa/ 
>
> `[2]` Mona Lisa Gif Evolution: https://github.com/peterbraden/genetic-lisa/blob/master/images/lisa-anim.gif 
>
> `[3]` Vase with Twelve Sunflowers (Arles, August 1888), Van Gogh. Neue Pinakothek, Munich: https://commons.wikimedia.org/wiki/File:Vincent_Willem_van_Gogh_128.jpg

