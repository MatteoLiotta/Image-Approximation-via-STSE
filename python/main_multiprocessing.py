## IMPORT LIBRARIES
import base64
import io
import os
import numpy as np
from PIL import Image
from PIL import ImageDraw
import aggdraw, random
import copy
from concurrent.futures import ProcessPoolExecutor
import time
import re
import argparse
import shutil
from tqdm import trange
from tqdm.notebook import tqdm


## EVOLUTION PARAMETERS


GENERATION_POPULATION = 10
CROSSOVER_POPULATION = 5
INITIAL_GENERATION = 10
NUM_PROCESSES = 8

EPOCHS = 50000                            #  <|===== CHOOSE
IMAGE_NUMBER = 6 # choose the picture     #  <|===== CHOOSE; default 1
RECOVER = 0                               #  <|===== CHOOSE; default 0
SHAPE_REDUCTION_RATE = 30                 #  <|===== CHOOSE; default (2)         Responsible for shape size. The higher, the smaller
MINMAX_SHAPE_MUT = (-10,10)               #  <|===== CHOOSE; default (-10,10)

USE_SHAPE_SIZE_SCHEDULING = 0

### ====================== DEFINE LOADING/SAVING PATHS ====================== ###

## Project main picture
if IMAGE_NUMBER == 1:
    image_path = "references/parisimage.JPG"
    save_path = "results/patches"
    w = 2048
    h = 3072
    reduce_ratio = 12
    split_rows=4
    split_cols=2

    REFERENCE = Image.open(image_path).resize((w//reduce_ratio, h//reduce_ratio))
    REFERENCE_DATA = REFERENCE.getdata()
    REFERENCE = REFERENCE.convert("RGB").quantize(colors=128)
    IMAGE_SIZE = REFERENCE.size


## Van gogh sunflowers
elif IMAGE_NUMBER == 2:
    image_path = "references/sunflower.jpg"
    save_path = "other_examples/sunflower"
    w = 192
    h = 240
    reduce_ratio = 1
    split_rows=4
    split_cols=2

    REFERENCE = Image.open(image_path).resize((w//reduce_ratio, h//reduce_ratio))
    REFERENCE_DATA = REFERENCE.getdata()
    REFERENCE = REFERENCE.convert("RGB").quantize(colors=128)
    IMAGE_SIZE = REFERENCE.size


## Black and white Trieste picture
elif IMAGE_NUMBER == 3:
    image_path = "references/trieste.JPG"
    save_path = "other_examples/trieste"
    w = 3130
    h = 2075
    reduce_ratio = 7
    split_rows=2
    split_cols=4

    REFERENCE = Image.open(image_path).resize((w//reduce_ratio, h//reduce_ratio))
    REFERENCE_DATA = REFERENCE.getdata()
    REFERENCE = REFERENCE.convert("RGB").quantize(colors=128)
    IMAGE_SIZE = REFERENCE.size

## Dalì
elif IMAGE_NUMBER == 4:
    image_path = "references/memory.jpg"
    save_path = "other_examples/memory"

    w = 330
    h = 240
    reduce_ratio = 1
    split_rows=2
    split_cols=4

    REFERENCE = Image.open(image_path).resize((w//reduce_ratio, h//reduce_ratio))
    REFERENCE_DATA = REFERENCE.getdata()
    REFERENCE = REFERENCE.convert("RGB").quantize(colors=128)
    IMAGE_SIZE = REFERENCE.size

if IMAGE_NUMBER == 5:
    image_path = "references/parisimage.JPG"
    save_path = "other_examples/smallshapeparis"

    NUM_PROCESSES = 12
    w = 2048
    h = 3072
    reduce_ratio = 8
    split_rows=4
    split_cols=3

    REFERENCE = Image.open(image_path).resize((w//reduce_ratio, h//reduce_ratio))
    REFERENCE_DATA = REFERENCE.getdata()
    REFERENCE = REFERENCE.convert("RGB").quantize(colors=128)
    IMAGE_SIZE = REFERENCE.size

if IMAGE_NUMBER == 6:
    NUM_PROCESSES = 12
    image_path = "references/sunflower.jpg"
    save_path = "other_examples/sunflower"
    w = 192
    h = 240
    reduce_ratio = 1
    split_rows=4
    split_cols=3

    REFERENCE = Image.open(image_path).resize((w//reduce_ratio, h//reduce_ratio))
    REFERENCE_DATA = REFERENCE.getdata()
    REFERENCE = REFERENCE.convert("RGB").quantize(colors=128)
    IMAGE_SIZE = REFERENCE.size

### ========================================================================= ###

## PREPARE THE FOLDER

def clear_folder(path):

    folder = path
    if os.path.exists(folder):
        shutil.rmtree(folder)
    os.makedirs(folder, exist_ok=True)




## SHAPE AND COLOR GENERATION


def generate_ellipse(shape_reduction_rate = SHAPE_REDUCTION_RATE):
    # generate two random numbers, which at maximum could be
    # as big as the image width (x) and height (y). 
    x = random.randint(0, IMAGE_SIZE[0])
    y = random.randint(0, IMAGE_SIZE[1])

    x_half_bound = IMAGE_SIZE[0] // 2
    y_half_bound = IMAGE_SIZE[1] // 2

    # Then the image width and image height
    max_w = IMAGE_SIZE[0] // shape_reduction_rate
    max_h = IMAGE_SIZE[1] // shape_reduction_rate

    w = random.randint(5, max_w)
    h = random.randint(5, max_h)

    # Calculate bounds
    xs = min(x, IMAGE_SIZE[0])       # left (the x)
    xr = min(x + w, IMAGE_SIZE[0])   # right (the x+w)
    yu = min(y, IMAGE_SIZE[1])       # up (the y)
    yd = min(y + h, IMAGE_SIZE[1])   # down (the y + h)

    if x > x_half_bound:
        x1 = max(x - w, 0)
        x2 = x
    else:
        x1 = x
        x2 = min(x + w, IMAGE_SIZE[0])

    if y > y_half_bound:
        y1 = max(y - h, 0)
        y2 = y
    else:
        y1 = y
        y2 = min(y + h, IMAGE_SIZE[1])

    # Return ellipse as (left, top, right, bottom)
    x1, x2 = sorted([x1, x2])
    y1, y2 = sorted([y1, y2])

    return "ellipse", [x1, y1, x2, y2]

def generate_rectangle(shape_reduction_rate = SHAPE_REDUCTION_RATE):
    x = random.randint(0, IMAGE_SIZE[0])
    y = random.randint(0, IMAGE_SIZE[1])
    w = random.randint(0, IMAGE_SIZE[0])
    h = random.randint(0, IMAGE_SIZE[1])

    x_half_bound = IMAGE_SIZE[0]//2
    y_half_bound = IMAGE_SIZE[1]//2

    # Calculate bounds
    max_w = IMAGE_SIZE[0] // shape_reduction_rate
    max_h = IMAGE_SIZE[1] // shape_reduction_rate

    w = random.randint(5, max_w)
    h = random.randint(5, max_h)

    if x > x_half_bound:
        x1 = max(x - w, 0)
        x2 = x
    else:
        x1 = x
        x2 = min(x + w, IMAGE_SIZE[0])

    if y > y_half_bound:
        y1 = max(y - h, 0)
        y2 = y
    else:
        y1 = y
        y2 = min(y + h, IMAGE_SIZE[1])

    x1, x2 = sorted([x1, x2])
    y1, y2 = sorted([y1, y2])

    return "rectangle", [x1, y1, x2, y2]

def generate_triangle(shape_reduction_rate = SHAPE_REDUCTION_RATE):
    points = []
    x_half_bound = IMAGE_SIZE[0] // 2
    y_half_bound = IMAGE_SIZE[1] // 2

    # Center
    cx = random.randint(0, IMAGE_SIZE[0])
    cy = random.randint(0, IMAGE_SIZE[1])

    max_dx = IMAGE_SIZE[0] // shape_reduction_rate
    max_dy = IMAGE_SIZE[1] // shape_reduction_rate

    for _ in range(3):
        dx = random.randint(-max_dx, max_dx)
        dy = random.randint(-max_dy, max_dy)

        px = min(max(cx + dx, 0), IMAGE_SIZE[0])
        py = min(max(cy + dy, 0), IMAGE_SIZE[1])

        points.extend([px, py])

    return "triangle", points

def generate_color(prev = None, a_mutation = (-0x5,0x3), rgb_mutation = (-0x5,0x3)):
	# Wheter the previous color is passed, the result will be
	# - The previous with a random variation within a limited scale 
	# - A new randomic color 
	if prev:
		# the change range wrt the previous is +-3 for the color red channel 
		# and +- 5 for color tonality and transparency
		return (prev[0] + random.randint(*a_mutation),  # R
		  	    prev[1] + random.randint(*rgb_mutation),  # G
				prev[2] + random.randint(*rgb_mutation),  # B
				prev[3] + random.randint(*rgb_mutation))  # A
	
	return (random.randint(0x33, 0x99),  # R: between 51 and 153 -> this limits red colors to the excess
		    random.randint(0, 0xff),     # G: between 0 and 255
			random.randint(0, 0xff),     # B: between 0 and 255
			random.randint(0, 0xff))     # A: between 0 and 255

def generate_shape(
                    func_list = [generate_ellipse, generate_rectangle, generate_triangle],
                    prev = None,
                    position_change_min_max = (-5,5)
                  ):
    
    if prev == None:
        # extract randomly a shape to do
        generator_shape = random.choice(func_list)
        shape_type, coords = generator_shape()
        
        return {'type': shape_type, 'coords': coords}

    else:
        # Just change the position if the previous shape is given
        prev["coords"] = [i + random.randint(*position_change_min_max) for i in prev["coords"]]
        return prev



## INDIVIDUALS CLASS
#  Differences:
#               - The reference is saved since the patch is changing
#               - The fitness function

class Strain(object):

    '''
    This class generates individuals with their own dna, name and parents. The idea is
    to provide each individuals with their dna, which is an actual list of shapes to be applied
    in the canvas. 

    The dna of a Strain object is an actual list of shapes as:

        [ ... , {'shape': {'type': 'rectangle', 'coords': [221, 62, 348, 172]}, 'color': (137, 116, 58, 225)}, ... ]

    '''    

    ### INDIVIDUAL DEFINITION
    def __init__(self, 
                 name = None, # something like "genXX-N"
                 dna_string =  None, 
                 parents = None,
                 reference_image = None
                 ):

        self.reference_image = reference_image
        self._fitness = None

        ### If a dna string is given as input

        if dna_string: 
            d = dna_string
            self.dna = d['dna'] # save it
            self.name = d['name']
            self._fitness = d.get('fitness')

        ### Otherwise if WE HAVE parents...

        elif parents: # Breed randomly from parents
            
            # If no string is givem we can copy the dna of parents and
            # we can apply mutation over it and crossover. 

            self.dna = copy.deepcopy(random.choice(parents).dna) # copy parents dna
            self.name = name # use this name
    
            # then apply the mutation: 
            # - if the random number (between 0 and 1 uniformly) is over 0.2 
            # - or if the dna length is not so elevated (so for short DNAs we "force" mutation
            #   to avoid same-dnas)
            if random.random() > 0.2 or len(self.dna)<20:
                self.mutate() # apply mutation
            
            # Now we need to do the CROSSOVER, but it makes sense only if the
            # individuals are not an empty list:
            
            if self.dna: # equivalent to "if self.dna != []:"

                for i in range(random.randint(1, len(parents))): # for each element of the dna string
                    # Randomly extract a shape from the a parent dna.
                    x = random.choice(parents).dna 
                    
                    y = x and random.choice(x) 
                    # In y, since the x is a entire dna list and the other is an extracted shape, the result will be
                    # - the specific extracted if it's not []
                    # - [] otherwise

                    # So, if the y is an actual gene and not a [], we need to produce an
                    # independent copy of that gene
                    y = y and copy.deepcopy(y)
                    
                    # then we extract an index number to point in the current individual 
                    # dna and... if its [] the gene remains the same, while otherwise it's 
                    # changed with the other above.
                    z = random.randint(0,len(self.dna)-1)
                    self.dna[z] = y or self.dna[z]
            
        ### BUT if we do NOT have parents...
        else:
            self.name = name
            self.dna = [] # the dna is initialized together with the individual name.


    ### MUTATION DEFINITION
    def mutate(self):
        # The numation is about changing the dna of the individuals

        x = random.random() # between 0 and 1, this will be analogous to mutation probability

        ## ADD A SHAPE
        #  with a high (75%) probability or with a not extremely large genome, 
        #  we can add a new gene shape
        if x<0.25 and len(self.dna)<500: 
            # PREVIOUS: self.dna.append({'ellipse' : generate_ellipse(), 'color' : generate_color()})
            self.dna.append({'shape' : generate_shape(), 'color' : generate_color()})
        
        ## CHANGE A SHAPE
        #  With the other 25% probability or with eventually big genomes we can 
        #  change a gene by popping randomly one of them.
        elif x<0.75: 
            # If the genome is too large 
            
            if self.dna: 
                # ... pop one 
                extracted_shape = self.dna.pop(random.randint(0,len(self.dna)-1))
                
                self.dna.append({'shape' : generate_shape(prev = extracted_shape["shape"], position_change_min_max=MINMAX_SHAPE_MUT), 
                                 'color' : generate_color(prev=extracted_shape["color"], a_mutation=(-30,30), rgb_mutation = (-40,40))})
                
            self.dna.append({'shape' : generate_shape(), 'color' : generate_color()}) # if no dna, no pop, just add

            ## FASTEST alternative
            idx = random.randint(0, len(self.dna) - 1)
            self.dna[idx], self.dna[-1] = self.dna[-1], self.dna[idx]  # swap with last one
            self.dna.pop()

        ## REMOVE A SHAPE
        #  This could only occur if the genome length is OVER max_length
        #  The maximum genome length is fixed.
        else:
            # or we can just pop
            if self.dna: self.dna.pop(random.randint(0,len(self.dna)-1))
        
        ## RECURSIVE MUTATION
        #  Since otherwise the mutation would occur on just one gene, recursive mutation
        #  admit with low probability also the case of whole dna change (specifically in the 
        #  evolution beginning phase)
        if random.random()<0.1:
            self.mutate()
        
        # Since the genome has been changed, the previously computed fitness stored 
        # is no more valid and we need to re-evaluate it.
        self._fitness = None    

    def fitness(self):
        return self._fitness or self._fitness_func()

    def _fitness_func(self):
        ref_img = self.reference_image.convert("RGBA")
        draw_img = np.array(self.draw().convert("RGBA"), dtype=np.int16)
        ref_img = np.array(ref_img, dtype=np.int16)

        if draw_img.shape != ref_img.shape:
            raise ValueError(f"Shape mismatch: {draw_img.shape} vs {ref_img.shape}")

        diff = draw_img[:, :, :3] - ref_img[:, :, :3]
        fitness = np.sum(np.abs(diff))
        self._fitness = fitness
        return fitness

    ## IMAGE-SHAPE DRAWING
    def draw(self):
        img = Image.new('RGBA', self.reference_image.size, (0, 0, 0, 0))  # canvas vuoto
        draw = aggdraw.Draw(img)

        for poly in self.dna:
            # in the dna are contained different shape informations we use
            # to fill the defined canvas

            shape = poly['shape'] # the shape name
            brush = aggdraw.Brush(poly['color'])

            # It clearly depends on the shapes the individual receives in the dna.
            # We can handle different cases, but they must be present in the global 
            # function defined for shape generation: generate_shape()
            
            if shape['type'] == 'ellipse':
                draw.ellipse(shape['coords'], brush)

            elif shape['type'] == 'rectangle':
                draw.rectangle(shape['coords'], brush)

            elif shape['type'] == 'triangle':
                draw.polygon(shape['coords'], brush)

        draw.flush()
        return img
    
    ## Individual representation
    def __repr__(self):
        d = {'name' : self.name, 'dna' : self.dna}
        if self._fitness:
            d['fitness'] = self._fitness
        return repr(d)



## PATCH HANDLING

def split_image_in_patches(image, rows = split_rows, cols = split_cols):
    # Split the image in a new collection of patches, by selecting the number of cuts
    # to do orizontally (row) and vertically (cols)    
    width, height = image.size
    patch_width = width // cols
    patch_height = height // rows

    patches = []
    idx = 0

    # Then we need coordinates to know where to cut patches
    for r in range(rows):
        for c in range(cols):
            left = c * patch_width
            upper = r * patch_height
            right = left + patch_width
            lower = upper + patch_height

            # So the current patch can be obtained cropping
            # and it's added to the collection, together with the reference (index)
            # to recover the full image at the end.
            patch_img = image.crop((left, upper, right, lower))
            patches.append({
                'index': idx,
                'row': r,
                'col': c,
                'pos': (left, upper),
                'patch': patch_img
            })
            idx += 1
    return patches



## RESTART EVOLUTION

def recover_population(patch_num, base_folder=save_path):
    folder = os.path.join(base_folder, f"num_{patch_num}")
    #print(f"\n{folder}\n")
    if not os.path.exists(folder):
        print(f"Folder {folder} does not exist. Created.")
        return None, 0  # nessun dato e gen 0

    # Cerca i file gen-*.txt e trova l'ultima generazione salvata
    saved_files = [f for f in os.listdir(folder) if f.startswith("gen-") and f.endswith(".txt")]
    if not saved_files:
        print(f"No saved generations found in {folder}.")
        return None, 0

    # Estrai numeri generazione e trova la più alta
    generations = []
    for f in saved_files:
        match = re.search(r"gen-(\d+)\.txt", f)
        if match:
            generations.append(int(match.group(1)))
    if not generations:
        return None, 0

    latest_gen = max(generations)
    path = os.path.join(folder, f"gen-{latest_gen}.txt")

    try:
        with open(path, "r") as f:
            content = f.read()
        namespace = {}
        exec(content, {}, namespace)
        initial_data = [Strain(dna_string=x) for x in namespace['initial_data']]
        print(f"Recovered patch {patch_num} from generation {latest_gen}")
        return initial_data, latest_gen
    except Exception as e:
        print(f"Error recovering patch {patch_num}:", e)
        return None, 0



## EVOLUTION WITH PATCHES
#  Unchanged

def evolution_patches(
    epochs_number=10, 
    initial_data=None, 
    i=0, 
    #fitnesses = [],
    processes_number=0,
    reference_patch=None  # patch di riferimento per questo processes
):
    if initial_data is None:
        initial_data = [Strain(name='initial', reference_image=reference_patch) for _ in range(CROSSOVER_POPULATION)]

    best = 999999999999999999
    local_SHAPE_REDUCTION_RATE = SHAPE_REDUCTION_RATE # can be used for scheduling

    crossover_strains = initial_data
    last_img = best
    last_best_epoch = 0
    epoch = 0

    #pbar = tqdm(total=epochs_number, desc="Evolution")

    while epoch < epochs_number:

        i += 1
        epoch += 1
        local_SHAPE_REDUCTION_RATE = local_SHAPE_REDUCTION_RATE * 1/i
        generation = []
        for j in range(GENERATION_POPULATION):
            p = Strain(parents=crossover_strains, name=f"gen{i}-{j}", reference_image=reference_patch)
            generation.append(p)

        ### SELECTION
        # Keep previous best of breed:
        generation.extend(crossover_strains)
        generation.sort(key = lambda x: x.fitness())

        crossover_strains = generation[:CROSSOVER_POPULATION]
        best = crossover_strains[0].fitness()
        #fitnesses.append(best)
        # del(generation)

        if i % 100 == 0:
            if processes_number == 11: 
                print(i)
            crossover_strains[0].draw().convert("RGB").save(f"{save_path}/num_{processes_number}/gen-{i}.png")
            with open(f"{save_path}/num_{processes_number}/gen-{i}.txt", "w") as save:
                save.write("initial_data = " + repr(crossover_strains))
        
        #pbar.update(1)
        #pbar.set_postfix({"last_best_epoch": last_best_epoch, "epoch": epoch})

    #pbar.close()
    return crossover_strains 



## MULTIPROCESSING FUNCTIONS

def run_patch(processes_num):
    ## PROCESS SPECIFIC
    # Division in the number of patches selected: processes_num
    patches = split_image_in_patches(REFERENCE, rows=split_rows, cols=split_cols)

    patch_info = patches[processes_num] 
    patch_img = patch_info['patch']

    # Recover previous individuals (eventually) 
    initial_data, initial_gen = recover_population(processes_num)
    # But otherwise we can initialize as always
    if initial_data is None:
        initial_data = [Strain(name=f'initial_{processes_num}', reference_image=patch_img) for i in range(CROSSOVER_POPULATION)]
        # NOTICE that the only difference is that we have to pass a reference image which process dependent
        initial_gen = 0
    else:
        # update the image given
        for strain in initial_data:
            strain.reference_image = patch_img

    # The return is the actual population
    return evolution_patches(
        epochs_number=EPOCHS,
        initial_data=initial_data,
        i=initial_gen,
        processes_number=processes_num,
        reference_patch=patch_img
    )


def run_all(num_PROCESSES=NUM_PROCESSES):
    # RUN ALL PROCESSES
    with ProcessPoolExecutor(max_workers=num_PROCESSES) as executor:
        results = list(executor.map(run_patch, range(num_PROCESSES)))
    return results

## START

if __name__ == "__main__":
    print(image_path, save_path)

    if RECOVER == False:
        clear_folder(save_path)

        for process in range(NUM_PROCESSES):
            folder_path = f"{save_path}/num_{process}"
            os.makedirs(folder_path, exist_ok=True)


    start_t = time.time()
    results = run_all(NUM_PROCESSES)
    end_t = time.time()
